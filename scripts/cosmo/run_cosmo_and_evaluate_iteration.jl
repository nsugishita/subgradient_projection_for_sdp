using FileIO, COSMO, SparseArrays, LinearAlgebra, Test, JuMP, MosekTools, Printf, NPZ, YAML

function evaluate_solution(input_file_path, problem_name, x)
    solution_path = "outputs/v2/cosmo/tmp/$(problem_name).npy"
    npzwrite(solution_path, x)
    res = readchomp(`bash -c ". ./scripts/activate.sh >/dev/null 2>&1 && python scripts/evaluate_solution.py --problem $(input_file_path) --solution $(solution_path)"`)
    f, f_gap, g = split(res, " ")
    f = parse(Float64, f)
    f_gap = parse(Float64, f_gap)
    g = parse(Float64, g)
    return f, f_gap, g
end

function optimize2!(ws::COSMO.Workspace{T}, input_file_path, problem_name, tol, warmup) where {T <: AbstractFloat}
    # >>>
    out = Dict(
        # "walltime" => [],
        "n_iterations" => [],
        # "x" => [],
        "obj" => [],
        "f" => [],
        "f_gap" => [],
        "g" => [],
        "r_prim" => [],
        "r_dual" => [],
        "max_norm_prim" => [],
        "max_norm_dual" => [],
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

        # convergence / infeasibility / timelimit checks
        cost, status, res_info = COSMO.check_termination!(ws, settings, iter, cost, status, res_info, time_limit_start, n)

        # >>>>
        ws2 = deepcopy(ws)
        COSMO.recover_μ!(ws2.vars.μ, ws2.vars.w_prev, ws2.vars.s, ws2.ρvec, n)
        if settings.scaling != 0
            COSMO.reverse_scaling!(ws2)
        end
        if ws2.ci.decompose
             COSMO.reverse_decomposition!(ws2, settings)
            y = -ws2.vars.μ
         else
            @. ws2.utility_vars.vec_m = -ws2.vars.μ
            y = ws2.utility_vars.vec_m
        end

        f, f_gap, g = evaluate_solution(input_file_path, problem_name, ws2.vars.x)

        push!(out["n_iterations"], iter + ws.safeguarding_iter )
        # push!(out["x"], ws2.vars.x)
        push!(out["obj"], cost)
        push!(out["f"], f)
        push!(out["f_gap"], f_gap)
        push!(out["g"], g)
        push!(out["r_prim"], res_info.r_prim)
        push!(out["r_dual"], res_info.r_dual)
        push!(out["max_norm_prim"], res_info.max_norm_prim)
        push!(out["max_norm_dual"], res_info.max_norm_dual)

        if warmup
            break
        end

        # NS New terminal condition
        @printf("%5d  f: %9.4f  g: %9.4f  r_p: %9.4f  r_d: %9.4f\n", iter + ws.safeguarding_iter, f_gap, g, res_info.r_prim, res_info.r_dual)
        if (f_gap <= tol) && (g <= 1e-3)
            break
        end
        # <<<<

        # >>>
        # NS Oirignal terminal condition
        # if status != :Undetermined
        #     break
        # end
        # <<<

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
    # println("x: $(size(ws.vars.x))  y: $(size(ws.vars.μ))   s: $(size(ws.vars.s))")
    COSMO.free_memory!(ws)
    return COSMO.Result{T}(ws.vars.x, y, ws.vars.s.data, cost, total_iter, ws.safeguarding_iter, status, res_info, ws.times), out;

end



function run_cosmo_and_evaluate_iteration(input_file_path, problem_name, tol, n_iterations, warmup)
    data = load(input_file_path);
    F = data["F"]
    c = data["c"]
    m = data["m"]
    n = data["n"]

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
    model = COSMO.Model();
    settings = COSMO.Settings(
         verbose = false,
         check_termination = 1,
         max_iter = n_iterations,
         eps_abs = 0,
         eps_rel= 0,
    )
    COSMO.assemble!(model, spzeros(m, m), c, cs1, settings = settings);
    # res = COSMO.optimize!(model);
    res, out = optimize2!(model, input_file_path, problem_name, tol, warmup);
    buf = Dict{String,Any}()
    for (key, value) in out
        buf[key] = stack(value, dims=1)
    end
    return buf;
end

input_file_paths = [
    "data/SDPLIB/data/mcp250-1.jld2",
    "data/SDPLIB/data/mcp250-2.jld2",
    "data/SDPLIB/data/mcp250-3.jld2",
    "data/SDPLIB/data/mcp250-4.jld2",
    "data/SDPLIB/data/mcp500-1.jld2",
    "data/SDPLIB/data/mcp500-2.jld2",
    "data/SDPLIB/data/mcp500-3.jld2",
    "data/SDPLIB/data/mcp500-4.jld2",
    "data/SDPLIB/data/gpp250-1.jld2",
    "data/SDPLIB/data/gpp250-2.jld2",
    "data/SDPLIB/data/gpp250-3.jld2",
    "data/SDPLIB/data/gpp250-4.jld2",
    "data/SDPLIB/data/gpp500-1.jld2",
    "data/SDPLIB/data/gpp500-2.jld2",
    "data/SDPLIB/data/gpp500-3.jld2",
    "data/SDPLIB/data/gpp500-4.jld2",
    "data/rudy/out/graph_1000_5_1.jld2",
    "data/rudy/out/graph_1000_5_2.jld2",
    "data/rudy/out/graph_1000_5_3.jld2",
    "data/rudy/out/graph_1000_5_4.jld2",
    "data/rudy/out/graph_2000_5_1.jld2",
    "data/rudy/out/graph_2000_5_2.jld2",
    "data/rudy/out/graph_2000_5_3.jld2",
    "data/rudy/out/graph_2000_5_4.jld2",
    "data/rudy/out/graph_3000_5_1.jld2",
    "data/rudy/out/graph_3000_5_2.jld2",
    "data/rudy/out/graph_3000_5_3.jld2",
    "data/rudy/out/graph_3000_5_4.jld2",
    "data/rudy/out/graph_4000_5_1.jld2",
    "data/rudy/out/graph_4000_5_2.jld2",
    "data/rudy/out/graph_4000_5_3.jld2",
    "data/rudy/out/graph_4000_5_4.jld2",
    "data/rudy/out/graph_5000_5_1.jld2",
    "data/rudy/out/graph_5000_5_2.jld2",
    "data/rudy/out/graph_5000_5_3.jld2",
    "data/rudy/out/graph_5000_5_4.jld2",
    "data/rudy/out/graph_1000_10_1.jld2",
    "data/rudy/out/graph_1000_10_2.jld2",
    "data/rudy/out/graph_1000_10_3.jld2",
    "data/rudy/out/graph_1000_10_4.jld2",
    "data/rudy/out/graph_2000_10_1.jld2",
    "data/rudy/out/graph_2000_10_2.jld2",
    "data/rudy/out/graph_2000_10_3.jld2",
    "data/rudy/out/graph_2000_10_4.jld2",
    "data/rudy/out/graph_3000_10_1.jld2",
    "data/rudy/out/graph_3000_10_2.jld2",
    "data/rudy/out/graph_3000_10_3.jld2",
    "data/rudy/out/graph_3000_10_4.jld2",
    "data/rudy/out/graph_4000_10_1.jld2",
    "data/rudy/out/graph_4000_10_2.jld2",
    "data/rudy/out/graph_4000_10_3.jld2",
    "data/rudy/out/graph_4000_10_4.jld2",
    "data/rudy/out/graph_5000_10_1.jld2",
    "data/rudy/out/graph_5000_10_2.jld2",
    "data/rudy/out/graph_5000_10_3.jld2",
    "data/rudy/out/graph_5000_10_4.jld2",
    "data/rudy/out/graph_1000_15_1.jld2",
    "data/rudy/out/graph_1000_15_2.jld2",
    "data/rudy/out/graph_1000_15_3.jld2",
    "data/rudy/out/graph_1000_15_4.jld2",
    "data/rudy/out/graph_2000_15_1.jld2",
    "data/rudy/out/graph_2000_15_2.jld2",
    "data/rudy/out/graph_2000_15_3.jld2",
    "data/rudy/out/graph_2000_15_4.jld2",
    "data/rudy/out/graph_3000_15_1.jld2",
    "data/rudy/out/graph_3000_15_2.jld2",
    "data/rudy/out/graph_3000_15_3.jld2",
    "data/rudy/out/graph_3000_15_4.jld2",
    "data/rudy/out/graph_4000_15_1.jld2",
    "data/rudy/out/graph_4000_15_2.jld2",
    "data/rudy/out/graph_4000_15_3.jld2",
    "data/rudy/out/graph_4000_15_4.jld2",
    "data/rudy/out/graph_5000_15_1.jld2",
    "data/rudy/out/graph_5000_15_2.jld2",
    "data/rudy/out/graph_5000_15_3.jld2",
    "data/rudy/out/graph_5000_15_4.jld2",
    "data/rudy/out/graph_1000_20_1.jld2",
    "data/rudy/out/graph_1000_20_2.jld2",
    "data/rudy/out/graph_1000_20_3.jld2",
    "data/rudy/out/graph_1000_20_4.jld2",
    "data/rudy/out/graph_2000_20_1.jld2",
    "data/rudy/out/graph_2000_20_2.jld2",
    "data/rudy/out/graph_2000_20_3.jld2",
    "data/rudy/out/graph_2000_20_4.jld2",
    "data/rudy/out/graph_3000_20_1.jld2",
    "data/rudy/out/graph_3000_20_2.jld2",
    "data/rudy/out/graph_3000_20_3.jld2",
    "data/rudy/out/graph_3000_20_4.jld2",
    "data/rudy/out/graph_4000_20_1.jld2",
    "data/rudy/out/graph_4000_20_2.jld2",
    "data/rudy/out/graph_4000_20_3.jld2",
    "data/rudy/out/graph_4000_20_4.jld2",
    "data/rudy/out/graph_5000_20_1.jld2",
    "data/rudy/out/graph_5000_20_2.jld2",
    "data/rudy/out/graph_5000_20_3.jld2",
    "data/rudy/out/graph_5000_20_4.jld2",
];


# input_file_paths = [
#     "data/SDPLIB/data/mcp100.jld2",
# ];

run_cosmo_and_evaluate_iteration("data/SDPLIB/data/mcp100.jld2", "mcp100", 1e-1, 10, true);
run_cosmo_and_evaluate_iteration("data/SDPLIB/data/mcp100.jld2", "mcp100", 1e-1, 10, true);

for input_file_path in input_file_paths
    problem_name = split(split(input_file_path, "/")[end], ".")[1];
    for tol in [1e-3]
        println("+++ solving $(problem_name) with tol $(tol)");
        long_run_path = "outputs/v2/cosmo/long_run/$(problem_name).txt";
        long_run = YAML.load_file(long_run_path);
        if (long_run["f_gap"] > tol) || (long_run["g"] > 1e-3)
            continue
        end
        result = run_cosmo_and_evaluate_iteration(input_file_path, problem_name, tol, long_run["n_iterations"], false);
        result_path = "outputs/v2/cosmo/iterations/$(problem_name).npz"
        npzwrite(result_path, result);
    end
end


# run_cosmo_and_evaluate_iteration("mcp100")
# run_cosmo_and_evaluate_iteration("mcp250-1")


# vimquickrun: julia --project=juliaenv examples/run_cosmo_and_evaluate_iteration.jl mcp250-1 mcp250-2
