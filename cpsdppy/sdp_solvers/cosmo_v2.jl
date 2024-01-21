using FileIO, COSMO, SparseArrays, LinearAlgebra, Test, JuMP, JSON, NPZ

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
    return model, JuMP.value.(x);
end

function run_cosmo_with_various_max_iters(problem_name, max_iter)
    data = load("./data/SDPLIB/data/$(problem_name).jld2");
    F = data["F"]
    c = data["c"]
    m = data["m"]
    n = data["n"]


    walltime_list = []
    n_iterations_list = []
    x_list = []
    obj_list = []

    for i in 1:max_iter
        println("iteration: $(i) / $(max_iter)");
        model = JuMP.Model(optimizer_with_attributes(
                COSMO.Optimizer, "verbose" => true, "max_iter" => i));
        @variable(model, x[1:m]);
        @objective(model, Min, c' * x);
        @constraint(model, con1,  Symmetric(-Matrix(F[1]) + sum(Matrix(F[k + 1]) .* x[k] for k in 1:m))  in JuMP.PSDCone());

        # set_attribute(model, "max_iter", i)
        JuMP.optimize!(model);
        results = backend(model).optimizer.model.optimizer.results

        push!(walltime_list, solve_time(model))
        push!(n_iterations_list, results.iter)
        push!(obj_list, JuMP.objective_value(model))
        push!(x_list, JuMP.value.(x))

        npzwrite(
            "cosmo_results/$(problem_name).npz",
            Dict(
                 "walltime" => Vector{Float64}(walltime_list),
                 "n_iterations" => Vector{Int64}(n_iterations_list),
                 "obj" => Vector{Float64}(obj_list),
                 "x" => stack(x_list, dims=1),
            )
        )
    end
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
    model, x = run_cosmo(problem_name, kwargs);
    results = backend(model).optimizer.model.optimizer.results

    npzwrite("tmp.npz", Dict("x" => x));

    println("walltime: $(solve_time(model))");
    println("n_iterations: $(results.iter)");

    max_iter = floor(Int64, 1.5 * results.iter)

    run_cosmo_with_various_max_iters(problem_name, max_iter)
end

run_experiments();
