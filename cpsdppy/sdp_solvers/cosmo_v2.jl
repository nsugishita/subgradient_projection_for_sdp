using FileIO, COSMO, SparseArrays, LinearAlgebra, Test, JuMP, JSON, NPZ, Printf

function run_cosmo_with_various_max_iters(problem_name)
    data = load("./data/SDPLIB/data/$(problem_name).jld2");
    F = data["F"]
    c = data["c"]
    m = data["m"]
    n = data["n"]

    solution_path = "cosmo_results/tmp_$(problem_name)_solution.npy"
    result_path = "cosmo_results/$(problem_name).npz"

    walltime_list = []
    n_iterations_list = []
    x_list = []
    obj_list = []
    f_list = []
    fgap_list = []
    g_list = []
    rprim_list = []
    rdual_list = []
    max_norm_prim_list = []
    max_norm_dual_list = []

    out = Dict(
        "walltime" => [],
        "n_iterations" => [],
        "x" => [],
        "obj" => [],
        "f" => [],
        "fgap" => [],
        "g" => [],
        "rprim" => [],
        "rdual" => [],
        "max_norm_prim" => [],
        "max_norm_dual" => [],
    )

    # To read the last result
    # TODO This does not work (when we save the results we get a type error).
    # if isfile(result_path)
    #     loaded = npzread(result_path)
    #     for (key, value) in out
    #         for x in loaded[key]
    #             push!(out[key], x)
    #         end
    #     end
    # end

    iteration = size(out["f"])[1] + 1

    while true
        model = JuMP.Model(optimizer_with_attributes(
                COSMO.Optimizer, "verbose" => false, "max_iter" => iteration));
        @variable(model, x[1:m]);
        @objective(model, Min, c' * x);
        @constraint(model, con1,  Symmetric(-Matrix(F[1]) + sum(Matrix(F[k + 1]) .* x[k] for k in 1:m))  in JuMP.PSDCone());

        # set_attribute(model, "max_iter", iteration)
        JuMP.optimize!(model);
        results = backend(model).optimizer.model.optimizer.results
        res_info = MOI.get(model, COSMO.RawResult()).info

        npzwrite(solution_path, JuMP.value.(x))

        res = readchomp(`bash -c ". ./scripts/activate.sh && python scripts/evaluate_solution.py --problem $(problem_name) --solution $(solution_path)"`)
        f, fgap, g = split(res, " ")
        f = parse(Float64, f)
        fgap = parse(Float64, fgap)
        g = parse(Float64, g)

        push!(out["walltime"], solve_time(model))
        push!(out["n_iterations"], results.iter)
        push!(out["obj"], JuMP.objective_value(model))
        push!(out["x"], JuMP.value.(x))
        push!(out["f"], f)
        push!(out["fgap"], fgap)
        push!(out["g"], g)
        push!(out["rprim"], res_info.r_prim)
        push!(out["rdual"], res_info.r_dual)
        push!(out["max_norm_prim"], res_info.max_norm_prim)
        push!(out["max_norm_dual"], res_info.max_norm_dual)

        open("cosmo_results/$(problem_name)_out.txt","a") do io
            @printf(io, "problem: %8s  iter: %5d  f: %9.4f  g: %9.4f  time: %9.4f\n", problem_name, iteration, fgap, g, solve_time(model))
        end

        solved = (fgap <= 1e-3) && (g <= 1e-3)

        if (iteration % 5 == 0) || solved
            buf = Dict{String,Any}()
            for (key, value) in out
                buf[key] = stack(value, dims=1)
            end
            npzwrite(result_path, buf)
        end

        if solved
            break
        end
        iteration += 1
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

    open("cosmo_results/$(problem_name)_out.txt","a") do io
        println(io, "problem_name: $(problem_name)");
        println(io, "opt_tol: $(opt_tol)");
        println(io, "feas_tol: $(feas_tol)");
        println(io, "pid: $(getpid())");
    end

    run_cosmo_with_various_max_iters(problem_name)
end

run_experiments();

# vimquickrun: bash tmp.sh 7
