using FileIO, COSMO, SparseArrays, LinearAlgebra, Test, JuMP, JSON

function run_cosmo(problem_name, kwargs=Dict())

    data = load("./data/SDPLIB/data/$(problem_name).jld2");
    F = data["F"]
    c = data["c"]
    m = data["m"]
    n = data["n"]

    model = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, "verbose" => true, kwargs...));
    @variable(model, x[1:m]);
    @objective(model, Min, c' * x);
    @constraint(model, con1,  Symmetric(-Matrix(F[1]) + sum(Matrix(F[k + 1]) .* x[k] for k in 1:m))  in JuMP.PSDCone());
    JuMP.optimize!(model);
    return model;

end

function run_experiments()
    # Parse the command line arguments.
    if size(ARGS)[1] != 2
        println("usage: julia $(PROGRAM_FILE) PROBLEM_NAME TOL");
        exit(1)
    end

    problem_name = ARGS[1]
    opt_tol = tryparse(Float64, ARGS[2])
    if opt_tol === nothing
        println("failed to parse tolerance: $(ARGS[2])");
        exit(1)
    end

    feas_tol = 1e-3

    println("problem_name: $(problem_name)");
    println("opt_tol: $(opt_tol)");
    println("feas_tol: $(feas_tol)");

    # Run the solver to make sure the program is compiled.
    println("compiling...");

    run_cosmo("theta1", Dict("verbose"=>false));
    run_cosmo("theta1", Dict("verbose"=>false));

    println("running the solver");

    kwargs = Dict{String, Any}(
      "eps_rel" => opt_tol,
      "eps_prim_inf" => feas_tol,
      "eps_dual_inf" => feas_tol,
      "max_iter" => 10000000,
      "time_limit" => 3600.0,
      "verbose" => true,
    )
    model = run_cosmo(problem_name, kwargs);
    results = backend(model).optimizer.model.optimizer.results

    println("walltime: $(solve_time(model))");
    println("n_iterations: $(results.iter)");
end

run_experiments();
