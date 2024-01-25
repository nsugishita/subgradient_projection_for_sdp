using FileIO, COSMO, SparseArrays, LinearAlgebra, Test, JuMP, MosekTools, Printf, NPZ

function optimize_with_iteration_time!(ws::COSMO.Workspace{T}) where {T <: AbstractFloat}
    # >>>
    out = Dict(
        "walltime" => [],
    )
    # <<<

    !ws.states.IS_ASSEMBLED && throw(ErrorException("The model has to be assembled! / set! before optimize!() can be called."))

    # start timer
    solver_time_start = time()

    settings = ws.settings

    # perform chordal decomposition
    if settings.decompose
        if !ws.states.IS_CHORDAL_DECOMPOSED
            ws.times.graph_time = @elapsed COSMO.chordal_decomposition!(ws)
        elseif ws.ci.decompose
            COSMO.pre_allocate_variables!(ws)
        end
    end

    # create scaling variables
    # with scaling    -> uses mutable diagonal scaling matrices
    # without scaling -> uses identity matrices
    if !ws.states.IS_SCALED
        ws.sm = (settings.scaling > 0) ? COSMO.ScaleMatrices{T}(ws.p.model_size[1], ws.p.model_size[2]) : COSMO.ScaleMatrices{T}()
    end

    # we measure times always in Float64
    ws.times.factor_update_time = 0.
    ws.times.proj_time  = 0. #reset projection time
    ws.times.setup_time = @elapsed COSMO.setup!(ws);
    if settings.verbose_timing
        ws.times.update_time = 0.
        ws.times.accelerate_time = 0.
    end

    # instantiate variables
    status = :Undetermined
    cost = T(Inf)
    res_info = COSMO.ResultInfo(T(Inf), T(Inf), zero(T), zero(T), ws.rho_updates)
    iter = 0
    ws.safeguarding_iter = 0
    # print information about settings to the screen
    settings.verbose && COSMO.print_header(ws)
    time_limit_start = time()

    m, n = ws.p.model_size


    COSMO.allocate_loop_variables!(ws, m, n)

    # warm starting the operator variable
    @. ws.vars.w[1:n] = ws.vars.x[1:n]
    @. ws.vars.w[n+1:n+m] = one(T) / ws.ρvec * ws.vars.μ + ws.vars.s.data

    # change state of the workspace
    ws.states.IS_OPTIMIZED = true

    iter_start = time()

    # do one initialisation step to make ADMM iterates agree with standard ADMM
    COSMO.admm_x!(ws.vars.s, ws.ν, ws.s_tl, ws.ls, ws.sol, ws.vars.w, ws.kkt_solver, ws.p.q, ws.p.b, ws.ρvec, settings.sigma, m, n)
    COSMO.admm_w!(ws.vars.s, ws.x_tl, ws.s_tl, ws.vars.w, settings.alpha, m, n);

    while iter + ws.safeguarding_iter < settings.max_iter
        iter += 1

        COSMO.acceleration_pre!(ws.accelerator, ws, iter)

        if COSMO.update_suggested(ws.infeasibility_check_due, ws.accelerator)
            COSMO.recover_μ!(ws.vars.μ, ws.vars.w_prev, ws.vars.s, ws.ρvec, n) # μ_k kept in sync with s_k, w already updated to w_{k+1}
            @. ws.δy.data = ws.vars.μ
        end

        # ADMM steps
        @. ws.vars.w_prev = ws.vars.w
        ws.times.proj_time += COSMO.admm_z!(ws.vars.s, ws.vars.w, ws.p.C, n)
        COSMO.apply_rho_adaptation_rules!(ws.ρvec, ws.rho_updates, settings, iter, iter_start, ws.times, ws, n)
        COSMO.admm_x!(ws.vars.s, ws.ν, ws.s_tl, ws.ls, ws.sol, ws.vars.w, ws.kkt_solver, ws.p.q, ws.p.b, ws.ρvec,settings.sigma, m, n)
        COSMO.admm_w!(ws.vars.s, ws.x_tl, ws.s_tl, ws.vars.w, settings.alpha, m, n);

        COSMO.acceleration_post!(ws.accelerator, ws, iter)
        # @show(ws.vars.w)

        # >>>
        push!(out["walltime"], time() - iter_start)
        # <<<

        # convergence / infeasibility / timelimit checks
        cost, status, res_info = COSMO.check_termination!(ws, settings, iter, cost, status, res_info, time_limit_start, n)
        if status != :Undetermined
            break
        end

    end #END-ADMM-MAIN-LOOP

    COSMO.recover_μ!(ws.vars.μ, ws.vars.w_prev, ws.vars.s, ws.ρvec, n)

    ws.times.iter_time = (time() - iter_start)
    settings.verbose_timing && (ws.times.post_time = time())

    # calculate primal and dual residuals
    if iter + ws.safeguarding_iter == settings.max_iter
        res_info = COSMO.calculate_result_info!(ws)
        status = :Max_iter_reached
    end

    # reverse scaling for scaled feasible cases
    if settings.scaling != 0
        COSMO.reverse_scaling!(ws)
    end

    #reverse chordal decomposition
    if ws.ci.decompose
         COSMO.reverse_decomposition!(ws, settings)
        y = -ws.vars.μ
     else
        @. ws.utility_vars.vec_m = -ws.vars.μ
        y = ws.utility_vars.vec_m
    end

    ws.times.solver_time = time() - solver_time_start
    settings.verbose_timing && (ws.times.post_time = time() - ws.times.post_time)

    # print solution to screen
    total_iter = ws.safeguarding_iter + iter
    settings.verbose && COSMO.print_result(status, total_iter, ws.safeguarding_iter, cost, ws.times.solver_time, ws.settings.safeguard)

    # create result object
    COSMO.free_memory!(ws)
    # >>>
    return COSMO.Result{T}(ws.vars.x, y, ws.vars.s.data, cost, total_iter, ws.safeguarding_iter, status, res_info, ws.times), out;
    # <<<

end

if size(ARGS)[1] == 0
    println("usage: julia $(PROGRAM_FILE) PROBLEM_NAME ...");
    exit(1)
end

function run_cosmo_with_time_limit(data, time_limit)
    F = data["F"]
    c = data["c"]
    m = data["m"]
    n = data["n"]
    obj_true = data["optVal"]

    d = div(n * (n + 1), 2)
    A_man = zeros(d, m)
    # now manually create A and B
    for (i, Fi) in enumerate(F[2:end])
      t = zeros(d)
      COSMO.extract_upper_triangle!(Fi, t, sqrt(2))
      A_man[:, i] = t
    end

    b_man = zeros(d)
    COSMO.extract_upper_triangle!(-F[1], b_man, sqrt(2))

    cs1 = COSMO.Constraint(A_man, b_man, COSMO.PsdConeTriangle)
    model_direct = COSMO.Model();
    settings = COSMO.Settings(
         verbose = true,
         time_limit = time_limit,
         max_iter = 10000000,
         eps_abs = 1e-8,
         eps_rel= 1e-8,
    )
    COSMO.assemble!(model_direct, spzeros(m, m), c, cs1, settings = settings);
    res, out = optimize_with_iteration_time!(model_direct);

    if res.status == :Solved
        status_code = 0
    elseif res.status == :Unsolved
        status_code = 1
    elseif res.status == :Max_iter_reached
        status_code = 2
    elseif res.status == :Time_limit_reached
        status_code = 3
    elseif res.status == :Primal_infeasible
        status_code = 4
    elseif res.status == :Dual_infeasible
        status_code = 5
    else
        status_code = 6
    end

    buf = Dict(
        "x" => res.x,
        "f" => res.obj_val,
        "f_star" => obj_true,
        "f_gap" => (res.obj_val - obj_true) / abs(obj_true),
        "time" => res.times.solver_time,
        "iter" => res.iter,
        "r_prim" => res.info.r_prim,
        "r_dual" => res.info.r_dual,
        "max_norm_prim" => res.info.max_norm_prim,
        "max_norm_dual" => res.info.max_norm_dual,
        "status" => status_code,
    );

    for (key, value) in out
        buf[key] = stack(value, dims=1)
    end

    return buf;
end

run_cosmo_with_time_limit(load("./data/SDPLIB/data/mcp100.jld2"), 3);

for problem_name in ARGS
    println("solving $(problem_name)");
    data = load("./data/SDPLIB/data/$(problem_name).jld2");

    result_path = "cosmo_results/run_cosmo_for_5_mins/$(problem_name).npz"
    result = run_cosmo_with_time_limit(data, 300);
    npzwrite(result_path, result);
end


# vimquickrun: julia --project=juliaenv examples/run_cosmo_and_evaluate_iteration.jl mcp250-1 mcp250-2
