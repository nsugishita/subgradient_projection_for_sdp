# Run COSMO for the exact number of iterations to satisfy the stopping condition
#
# To run this scripts use the following command from the top of the project.
#
# ```
# julia --project=juliaenv scripts/cosmo/run_cosmo_with_iteration_limit.jl
# ```

using FileIO, COSMO, SparseArrays, LinearAlgebra, Test, Printf, NPZ

function run_cosmo(input_file_path, tol, verbose)
    data = load(input_file_path);
    F = data["F"]
    c = data["c"]
    m = data["m"]
    n = data["n"]

    d = div(n * (n + 1), 2)
    A_man = zeros(d, m)
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
         verbose = verbose,
         max_iter = 10000000,
         eps_abs = tol,
         eps_rel= tol,
    )
    COSMO.assemble!(model, spzeros(m, m), c, cs1, settings = settings);
    res = COSMO.optimize!(model);

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
        "tol" => tol,
        "primal_objective" => res.obj_val,
        "walltime" => res.times.solver_time,
        "n_iterations" => res.iter,
        "r_prim" => res.info.r_prim,
        "r_dual" => res.info.r_dual,
        "max_norm_prim" => res.info.max_norm_prim,
        "max_norm_dual" => res.info.max_norm_dual,
        "status" => status_code,
    );
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
    # "data/rudy/out/weighted_graph_1000_5_1.jld2",
    # "data/rudy/out/weighted_graph_1000_5_2.jld2",
    # "data/rudy/out/weighted_graph_1000_5_3.jld2",
    # "data/rudy/out/weighted_graph_1000_5_4.jld2",
    # "data/rudy/out/weighted_graph_2000_5_1.jld2",
    # "data/rudy/out/weighted_graph_2000_5_2.jld2",
    # "data/rudy/out/weighted_graph_2000_5_3.jld2",
    # "data/rudy/out/weighted_graph_2000_5_4.jld2",
    # "data/rudy/out/weighted_graph_3000_5_1.jld2",
    # "data/rudy/out/weighted_graph_3000_5_2.jld2",
    # "data/rudy/out/weighted_graph_3000_5_3.jld2",
    # "data/rudy/out/weighted_graph_3000_5_4.jld2",
    # "data/rudy/out/weighted_graph_4000_5_1.jld2",
    # "data/rudy/out/weighted_graph_4000_5_2.jld2",
    # "data/rudy/out/weighted_graph_4000_5_3.jld2",
    # "data/rudy/out/weighted_graph_4000_5_4.jld2",
    # "data/rudy/out/weighted_graph_5000_5_1.jld2",
    # "data/rudy/out/weighted_graph_5000_5_2.jld2",
    # "data/rudy/out/weighted_graph_5000_5_3.jld2",
    # "data/rudy/out/weighted_graph_5000_5_4.jld2",
    # "data/rudy/out/weighted_graph_1000_10_1.jld2",
    # "data/rudy/out/weighted_graph_1000_10_2.jld2",
    # "data/rudy/out/weighted_graph_1000_10_3.jld2",
    # "data/rudy/out/weighted_graph_1000_10_4.jld2",
    # "data/rudy/out/weighted_graph_2000_10_1.jld2",
    # "data/rudy/out/weighted_graph_2000_10_2.jld2",
    # "data/rudy/out/weighted_graph_2000_10_3.jld2",
    # "data/rudy/out/weighted_graph_2000_10_4.jld2",
    # "data/rudy/out/weighted_graph_3000_10_1.jld2",
    # "data/rudy/out/weighted_graph_3000_10_2.jld2",
    # "data/rudy/out/weighted_graph_3000_10_3.jld2",
    # "data/rudy/out/weighted_graph_3000_10_4.jld2",
    # "data/rudy/out/weighted_graph_4000_10_1.jld2",
    # "data/rudy/out/weighted_graph_4000_10_2.jld2",
    # "data/rudy/out/weighted_graph_4000_10_3.jld2",
    # "data/rudy/out/weighted_graph_4000_10_4.jld2",
    # "data/rudy/out/weighted_graph_5000_10_1.jld2",
    # "data/rudy/out/weighted_graph_5000_10_2.jld2",
    # "data/rudy/out/weighted_graph_5000_10_3.jld2",
    # "data/rudy/out/weighted_graph_5000_10_4.jld2",
    # "data/rudy/out/weighted_graph_1000_15_1.jld2",
    # "data/rudy/out/weighted_graph_1000_15_2.jld2",
    # "data/rudy/out/weighted_graph_1000_15_3.jld2",
    # "data/rudy/out/weighted_graph_1000_15_4.jld2",
    # "data/rudy/out/weighted_graph_2000_15_1.jld2",
    # "data/rudy/out/weighted_graph_2000_15_2.jld2",
    # "data/rudy/out/weighted_graph_2000_15_3.jld2",
    # "data/rudy/out/weighted_graph_2000_15_4.jld2",
    # "data/rudy/out/weighted_graph_3000_15_1.jld2",
    # "data/rudy/out/weighted_graph_3000_15_2.jld2",
    # "data/rudy/out/weighted_graph_3000_15_3.jld2",
    # "data/rudy/out/weighted_graph_3000_15_4.jld2",
    # "data/rudy/out/weighted_graph_4000_15_1.jld2",
    # "data/rudy/out/weighted_graph_4000_15_2.jld2",
    # "data/rudy/out/weighted_graph_4000_15_3.jld2",
    # "data/rudy/out/weighted_graph_4000_15_4.jld2",
    # "data/rudy/out/weighted_graph_5000_15_1.jld2",
    # "data/rudy/out/weighted_graph_5000_15_2.jld2",
    # "data/rudy/out/weighted_graph_5000_15_3.jld2",
    # "data/rudy/out/weighted_graph_5000_15_4.jld2",
    # "data/rudy/out/weighted_graph_1000_20_1.jld2",
    # "data/rudy/out/weighted_graph_1000_20_2.jld2",
    # "data/rudy/out/weighted_graph_1000_20_3.jld2",
    # "data/rudy/out/weighted_graph_1000_20_4.jld2",
    # "data/rudy/out/weighted_graph_2000_20_1.jld2",
    # "data/rudy/out/weighted_graph_2000_20_2.jld2",
    # "data/rudy/out/weighted_graph_2000_20_3.jld2",
    # "data/rudy/out/weighted_graph_2000_20_4.jld2",
    # "data/rudy/out/weighted_graph_3000_20_1.jld2",
    # "data/rudy/out/weighted_graph_3000_20_2.jld2",
    # "data/rudy/out/weighted_graph_3000_20_3.jld2",
    # "data/rudy/out/weighted_graph_3000_20_4.jld2",
    # "data/rudy/out/weighted_graph_4000_20_1.jld2",
    # "data/rudy/out/weighted_graph_4000_20_2.jld2",
    # "data/rudy/out/weighted_graph_4000_20_3.jld2",
    # "data/rudy/out/weighted_graph_4000_20_4.jld2",
    # "data/rudy/out/weighted_graph_5000_20_1.jld2",
    # "data/rudy/out/weighted_graph_5000_20_2.jld2",
    # "data/rudy/out/weighted_graph_5000_20_3.jld2",
    # "data/rudy/out/weighted_graph_5000_20_4.jld2",
];

# input_file_paths = ["data/SDPLIB/data/mcp100.jld2"];

tols = [0.01, 0.001];

# run_cosmo("data/SDPLIB/data/mcp100.jld2", 0.1, false);
# run_cosmo("data/SDPLIB/data/mcp100.jld2", 0.1, false);

mkpath("outputs/v2/cosmo_original")

for input_file_path in input_file_paths
    problem_name = split(split(input_file_path, "/")[end], ".")[1];
    for tol in tols
        # println("+++ problem: $(problem_name)  tol: $(tol)");
        iteration_file_path = "outputs/v2/cosmo/iterations/$(problem_name).npz";
        iteration_data = npzread(iteration_file_path)
        n_iterations = findfirst((iteration_data["f_gap"] .<= tol) .& (iteration_data["g"] .<= 1e-3));
        println("+++ problem: $(problem_name)  tol: $(tol)  n_iters: $(n_iterations)");
        output_file_path = "outputs/v2/cosmo/$(problem_name)_tol_$(tol).txt";
        # res = run_cosmo(input_file_path, tol, true);
        # io = open(output_file_path, "w");
        # for (key, value) in res
        #     write(io, "$(key): $(value)\n");
        #     println("$(key): $(value)");
        # end
        # close(io);
    end
end
