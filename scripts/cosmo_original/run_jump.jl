# Run COSMO with default stopping criteria
#
# To run this scripts use the following command from the top of the project.
#
# ```
# julia --project=juliaenv scripts/cosmo_original/run.jl path/to/file.dat-s
# ```

using COSMO, SparseArrays, LinearAlgebra, JuMP

include("../read_sdpa_file.jl");

function run_cosmo(input_file_path, tol, feas_tol, verbose)
    data = read_sdpa_file(input_file_path);
    F = data["F"];
    c = data["c"];
    m = data["m"];
    n = data["n"];
    n_blocks = data["n_blocks"];

    kwargs = Dict{String, Any}(
      "eps_rel" => tol,
      "eps_prim_inf" => feas_tol,
      "eps_dual_inf" => feas_tol,
      "max_iter" => 10000000,
      "verbose" => verbose,
    );

    model = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, kwargs...));
    @variable(model, x[1:m]);
    @objective(model, Min, c' * x);
    @constraint(model, [block_index = 1:n_blocks],
        Symmetric(
            -Matrix(F[block_index, 1])
            + sum(Matrix(F[block_index, k + 1]) .* x[k] for k in 1:m)
        ) in JuMP.PSDCone()
    );
    JuMP.optimize!(model);

    results = backend(model).optimizer.model.optimizer.results;
    res_info = MOI.get(model, COSMO.RawResult()).info;

    status = termination_status(model);
    if status == :OPTIMAL
        status_code = 0;
    elseif status == :OPTIMIE_NOT_CALLED
        status_code = 1;
    elseif status == :ITERATION_LIMIT
        status_code = 2;
    elseif status == :TIME_LIMIT
        status_code = 3;
    elseif status == :INFEASIBLE
        status_code = 4;
    elseif status == :DUAL_INFEASIBLE
        status_code = 5;
    else
        status_code = 6;
    end

    return Dict(
        "tol" => tol,
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

tols = [0.01, 0.001];
feas_tol = 1e-3;

run_cosmo("data/SDPLIB/data/mcp100.dat-s", 0.5, 0.5, false);
run_cosmo("data/SDPLIB/data/mcp100.dat-s", 0.5, 0.5, false);

mkpath("outputs/v2/cosmo_jump_original")

for input_file_path in ARGS
    problem_name = split(split(input_file_path, "/")[end], ".")[1];
    for tol in tols
        println("+++ problem: $(problem_name)  tol: $(tol)");
        res = run_cosmo(input_file_path, tol, feas_tol, true);
        output_file_path = "outputs/v2/cosmo_jump_original/$(problem_name)_tol_$(tol).txt";
        io = open(output_file_path, "w");
        for (key, value) in res
            write(io, "$(key): $(value)\n");
            println("$(key): $(value)");
        end
        close(io);
    end
end
