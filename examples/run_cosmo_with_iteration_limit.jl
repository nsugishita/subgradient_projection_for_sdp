using FileIO, COSMO, SparseArrays, LinearAlgebra, Test, JuMP, MosekTools, Printf, NPZ

function run_cosmo_with_iteration_limit(problem_name, tol, debug)
    data = load("./data/SDPLIB/data/$(problem_name).jld2");
    F = data["F"]
    c = data["c"]
    m = data["m"]
    n = data["n"]
    obj_true = data["optVal"]

    prev_res = npzread("cosmo_results/run_cosmo_and_evaluate_iteration/$(problem_name).npz");
    # prev_iter = prev_res["iter"]
    # prev_time = prev_res["time"]
    # prev_x = prev_res["x"]
    # println("problem: $(problem_name)   iter: $(prev_iter)  time: $(prev_time)");
    max_iter = findfirst((prev_res["f_gap"] .<= tol) .& (prev_res["g"] .<= tol))
    println("problem: $(problem_name)  tol: $(tol)   iter: $(max_iter)");

    if debug
        max_iter = 10
    end

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
    if !isnothing(max_iter)
        settings = COSMO.Settings(
             verbose = false,
             max_iter = max_iter,
             eps_abs = 0,
             eps_rel= 0,
        )
    else
        settings = COSMO.Settings(
             verbose = false,
             time_limit = 300,
             eps_abs = 0,
             eps_rel= 0,
        )
    end
    COSMO.assemble!(model_direct, spzeros(m, m), c, cs1, settings = settings);
    res = COSMO.optimize!(model_direct);

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

    return Dict(
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
end

run_cosmo_with_iteration_limit("mcp250-1", 1e-2, true);

for problem_name in ARGS
    for tol in [1e-2, 1e-3]
        println("solving $(problem_name)");
        result = run_cosmo_with_iteration_limit(problem_name, tol, false);
        result_path = "cosmo_results/run_cosmo_with_iteration_limit/$(problem_name)_$(tol).npz"
        npzwrite(result_path, result);
    end
end


# run_cosmo_and_evaluate_iteration("mcp100")
# run_cosmo_and_evaluate_iteration("mcp250-1")


# vimquickrun: julia --project=juliaenv examples/run_cosmo_with_iteration_limit.jl mcp250-1 mcp250-2
