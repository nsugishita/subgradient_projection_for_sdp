# Find the number of iterations COSMO needs to solve the problems
#
# To run this scripts use the following command from the top of the project.
#
# ```
# julia --project=juliaenv scripts/cosmo/find_n_iterations.jl
# ```

using FileIO, COSMO, SparseArrays, LinearAlgebra, Test, Printf, NPZ, JuMP, YAML, Printf
import MathOptInterface as MOI

function evaluate_solution(input_file_path, problem_name, x)
    solution_path = "outputs/v2/cosmo/find_n_iterations/$(problem_name)_solution.npy"
    npzwrite(solution_path, x)
    res = readchomp(`bash -c ". ./scripts/activate.sh >/dev/null 2>&1 && python scripts/evaluate_solution.py --problem $(input_file_path) --solution $(solution_path)"`)
    f, f_gap, g = split(res, " ")
    f = parse(Float64, f)
    f_gap = parse(Float64, f_gap)
    g = parse(Float64, g)
    return f, f_gap, g
end

function build_model(input_file_path)
    data = load(input_file_path);
    F = data["F"]
    c = data["c"]
    m = data["m"]
    n = data["n"]

    kwargs = Dict{String, Any}(
      "eps_rel" => 1e-6,
      "eps_prim_inf" => 1e-6,
      "eps_dual_inf" => 1e-6,
      "verbose" => false,
    )

    model = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, kwargs...));
    @variable(model, x[1:m]);
    @objective(model, Min, c' * x);
    @constraint(model, con1,  Symmetric(-Matrix(F[1]) + sum(Matrix(F[k + 1]) .* x[k] for k in 1:m))  in JuMP.PSDCone());
    # MOI.Utilities.attach_optimizer(model);
    # inner = unsafe_backend(model).inner;
    # COSMO.assemble!(inner, spzeros(m, m), c, cs1, settings = settings);
    # return inner;
    return model, x;
end

function extract_solver_results(input_file_path, problem_name, res)
    if full
        f, f_gap, g = evaluate_solution(input_file_path, problem_name, res.x)
    else
        f = NaN
        f_gap = NaN
        g = NaN
    end

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
        "tol" => 1e-6,
        "time_limit" => time_limit,
        "primal_objective" => res.obj_val,
        "walltime" => res.times.solver_time,
        "n_iterations" => res.iter,
        "r_prim" => res.info.r_prim,
        "r_dual" => res.info.r_dual,
        "max_norm_prim" => res.info.max_norm_prim,
        "max_norm_dual" => res.info.max_norm_dual,
        "status" => status_code,
        "f" => f,
        "f_gap" => f_gap,
        "g" => g,
    );
end

function extract_solver_results_jump(input_file_path, problem_name, model, x)
    results = backend(model).optimizer.model.optimizer.results
    res_info = MOI.get(model, COSMO.RawResult()).info

    f, f_gap, g = evaluate_solution(input_file_path, problem_name, JuMP.value.(x))

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
        # "time_limit" => 0,
        "primal_objective" => JuMP.objective_value(model),
        "walltime" => solve_time(model),
        "n_iterations" => results.iter,
        "r_prim" => res_info.r_prim,
        "r_dual" => res_info.r_dual,
        "max_norm_prim" => res_info.max_norm_prim,
        "max_norm_dual" => res_info.max_norm_dual,
        "status" => status_code,
        "f" => f,
        "f_gap" => f_gap,
        "g" => g,
    );
end

function find_n_iterations(input_file_path, problem_name, tol, feas_tol, n_iterations, full)
    lb = 0
    ub = n_iterations

    model, x = build_model(input_file_path)

    while ub - lb > 2
        n_iterations = trunc(Int, 0.5 * (lb + ub));
        JuMP.set_optimizer(model, COSMO.Optimizer);
        set_attribute(model, "max_iter", n_iterations);
        set_attribute(model, "eps_rel", 0.0);
        set_attribute(model, "eps_abs", 0.0);
        set_attribute(model, "eps_prim_inf", 0.0);
        set_attribute(model, "eps_dual_inf", 0.0);
        set_attribute(model, "verbose", false);
        JuMP.optimize!(model);
        res = extract_solver_results_jump(input_file_path, problem_name, model, x);
        if (res["f_gap"] <= 1.2 * tol) && (res["g"] <= 1.2 * feas_tol)
            ub = min(ub, res["n_iterations"])
        else
            lb = max(lb, res["n_iterations"])
        end

        @printf(
            "%15s  mode: 1  it: %4d  t: %6.1f  f: %8.5f  g: %8.5f  bnd: %4d - %4d\n",
            problem_name, n_iterations, res["walltime"], res["f_gap"], res["g"], lb, ub
        )
    end

    n_iterations = lb + 1;

    while true
        JuMP.set_optimizer(model, COSMO.Optimizer);
        set_attribute(model, "max_iter", n_iterations);
        set_attribute(model, "eps_rel", 0.0);
        set_attribute(model, "eps_abs", 0.0);
        set_attribute(model, "eps_prim_inf", 0.0);
        set_attribute(model, "eps_dual_inf", 0.0);
        set_attribute(model, "verbose", false);
        JuMP.optimize!(model);
        res = extract_solver_results_jump(input_file_path, problem_name, model, x);

        @printf(
            "%15s  mode: 2  it: %4d  t: %6.1f  f: %8.5f  g: %8.5f\n",
            problem_name, n_iterations, res["walltime"], res["f_gap"], res["g"]
        )

        if (res["f_gap"] <= tol) && (res["g"] <= feas_tol)
            return res;
        end
        n_iterations += 1;
    end
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

# find_n_iterations("data/SDPLIB/data/mcp100.jld2", "mcp100", 0.1, 10.0, 10, false);

mkpath("outputs/v2/cosmo/find_n_iterations")

for input_file_path in input_file_paths
    problem_name = split(split(input_file_path, "/")[end], ".")[1];
    for tol in [1e-2, 1e-3]
        long_run_path = "outputs/v2/cosmo/long_run/$(problem_name).txt";
        long_run = YAML.load_file(long_run_path);
        if (long_run["f_gap"] > tol) || (long_run["g"] > 1e-3)
            continue
        end
        n_iterations = long_run["n_iterations"]
        println("+++ problem: $(problem_name)  tol: $(tol)  ub on n_iterations: $(n_iterations)");
        output_file_path = "outputs/v2/cosmo/find_n_iterations/$(problem_name)_tol_$(tol).txt";
        res = find_n_iterations(input_file_path, problem_name, tol, 1e-3, n_iterations, true);
        io = open(output_file_path, "w");
        for (key, value) in res
            write(io, "$(key): $(value)\n");
            println("$(key): $(value)");
        end
        close(io);
        if res["f_gap"] <= 1e-3
            #  We found the number of iterations to achieve the tightest tol.
            break
        end
    end
end
