# Run COSMO with default stopping criteria
#
# To run this scripts use the following command from the top of the project.
#
# ```
# julia --project=juliaenv scripts/cosmo_original/run.jl
# ```

using FileIO, COSMO, SparseArrays, LinearAlgebra, Test, Printf, JuMP

function run_cosmo(input_file_path, tol, verbose)
    data = load(input_file_path);
    F = data["F"]
    c = data["c"]
    m = data["m"]
    n = data["n"]

    kwargs = Dict{String, Any}(
      "eps_rel" => tol,
      "eps_prim_inf" => tol,
      "eps_dual_inf" => tol,
      "max_iter" => 10000000,
      "verbose" => verbose,
    )

    model = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, kwargs...));
    @variable(model, x[1:m]);
    @objective(model, Min, c' * x);
    @constraint(model, con1,  Symmetric(-Matrix(F[1]) + sum(Matrix(F[k + 1]) .* x[k] for k in 1:m))  in JuMP.PSDCone());
    JuMP.optimize!(model);

    results = backend(model).optimizer.model.optimizer.results
    res_info = MOI.get(model, COSMO.RawResult()).info

    status = termination_status(model)
    if status == :OPTIMAL
        status_code = 0
    elseif status == :OPTIMIE_NOT_CALLED
        status_code = 1
    elseif status == :ITERATION_LIMIT
        status_code = 2
    elseif status == :TIME_LIMIT
        status_code = 3
    elseif status == :INFEASIBLE
        status_code = 4
    elseif status == :DUAL_INFEASIBLE
        status_code = 5
    else
        status_code = 6
    end

    return Dict(
        "tol" => 1e-6,
        "time_limit" => 0,
        "primal_objective" => JuMP.objective_value(model),
        "walltime" => solve_time(model),
        "n_iterations" => results.iter,
        "r_prim" => res_info.r_prim,
        "r_dual" => res_info.r_dual,
        "max_norm_prim" => res_info.max_norm_prim,
        "max_norm_dual" => res_info.max_norm_dual,
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

# input_file_paths = ["data/SDPLIB/data/mcp100.jld2"];

tols = [0.01, 0.001];

run_cosmo("data/SDPLIB/data/mcp100.jld2", 0.1, false);
run_cosmo("data/SDPLIB/data/mcp100.jld2", 0.1, false);

mkpath("outputs/v2/cosmo_jump_original")

for input_file_path in input_file_paths
    problem_name = split(split(input_file_path, "/")[end], ".")[1];
    for tol in tols
        println("+++ problem: $(problem_name)  tol: $(tol)");
        output_file_path = "outputs/v2/cosmo_jump_original/$(problem_name)_tol_$(tol).txt";
        res = run_cosmo(input_file_path, tol, true);
        io = open(output_file_path, "w");
        for (key, value) in res
            write(io, "$(key): $(value)\n");
            println("$(key): $(value)");
        end
        close(io);
    end
end
