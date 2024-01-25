using FileIO, COSMO, SparseArrays, LinearAlgebra, Test, JuMP, MosekTools, NPZ

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

run_cosmo_with_time_limit(load("./data/SDPLIB/data/mcp100.jld2"), 3);

for problem_name in ARGS
    println("solving $(problem_name)");
    data = load("./data/SDPLIB/data/$(problem_name).jld2");

    result_path = "cosmo_results/run_cosmo_for_5_mins/$(problem_name).npz"
    result = run_cosmo_with_time_limit(data, 300);
    npzwrite(result_path, result);
end


# vimquickrun: julia --project=juliaenv examples/run_cosmo_for_5mins.jl mcp100
