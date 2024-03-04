# -*- coding: utf-8 -*-

"""Run SDPNAL+

This runs SDPNAL+ on a specified SDPLIB instance.
Since we do not know how many iterations we need to run unti we find a solution
satisfying the terminal conditions, we run SDPNAL+ repeatedly with various
number of iterations, until we find the required number of iterations.
"""

import sys
import os
import subprocess

import numpy as np

from cpsdppy import config as config_module
from cpsdppy import logging_helper, sdpa
from cpsdppy.sdp_solvers import common


def run(data_file_path, tol, feas_tol):
    cwd = os.getcwd()
    problem_name = os.path.splitext(os.path.basename(data_file_path))[0]
    output_file_name = (
        f"outputs/v3/sdpnal/{problem_name}_{tol}_{feas_tol}.npz"
    )

    if os.path.exists(output_file_name):
        return np.load(output_file_name)

    output_dir = os.path.dirname(output_file_name)

    solution_path = (
        f"{cwd}/{output_dir}/"
        f"{problem_name}_{tol}_{feas_tol}_tmp_sol.txt"
    )

    os.makedirs(output_dir, exist_ok=True)

    iteration_limit = 100

    ub = None
    lb = 1

    # Find ub
    while True:
        command = (
            f"bash scripts/internal/sdpnal/interface.sh {data_file_path} "
            f"{iteration_limit} {solution_path}"
        )
        res = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            encoding="utf8",
        )
        x = np.loadtxt(solution_path)

        config = config_module.Config()
        config.problem_name = data_file_path
        problem_data = sdpa.read(config)

        eval_x = common.evaluate_solution(x, problem_data)
        lines = res.stdout.split("\n")
        for line in lines:
            if "Computing time (total)" in line:
                pos = line.index("=") + 1
                walltime = float(line[pos:])

        print(
            f"{problem_name}  mode: 0  it: {iteration_limit:4d}  "
            f"t: {walltime:5.1f}  f: {eval_x.f_gap:8.5f}  "
            f"g: {eval_x.g.item():8.5f}"
        )

        if eval_x.f_gap <= 1.2 * tol and eval_x.g.item() <= 1.2 * feas_tol:
            ub = iteration_limit
            break

        iteration_limit += 100

    while True:
        iteration_limit = int((lb + ub) / 2)
        command = (
            f"bash scripts/internal/sdpnal/interface.sh {data_file_path} "
            f"{iteration_limit} {solution_path}"
        )
        res = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            encoding="utf8",
        )
        x = np.loadtxt(solution_path)

        config = config_module.Config()
        config.problem_name = data_file_path
        problem_data = sdpa.read(config)

        eval_x = common.evaluate_solution(x, problem_data)
        lines = res.stdout.split("\n")
        for line in lines:
            if "Computing time (total)" in line:
                pos = line.index("=") + 1
                walltime = float(line[pos:])

        if eval_x.f_gap <= 1.2 * tol and eval_x.g.item() <= 1.2 * feas_tol:
            ub = min(ub, iteration_limit)
            if ub < lb:
                lb = 1
        else:
            lb = max(lb, iteration_limit)

        print(
            f"{problem_name}  mode: 1  it: {iteration_limit:4d}  "
            f"t: {walltime:5.1f}  f: {eval_x.f_gap:8.5f}  "
            f"g: {eval_x.g.item():8.5f}  bnd: {lb:4d} - {ub:4d}"
        )

        if ub - lb <= 2:
            break

    iteration_limit = lb + 1

    out = {
        "n_iterations": [],
        "walltime": [],
        "f": [],
        "f_gap": [],
        "g": [],
        "x": [],
    }

    while True:
        command = (
            f"bash scripts/internal/sdpnal/interface.sh {data_file_path} "
            f"{iteration_limit} {solution_path}"
        )
        res = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            encoding="utf8",
        )
        x = np.loadtxt(solution_path)

        config = config_module.Config()
        config.problem_name = data_file_path
        problem_data = sdpa.read(config)

        eval_x = common.evaluate_solution(x, problem_data)
        lines = res.stdout.split("\n")
        for line in lines:
            if "Computing time (total)" in line:
                pos = line.index("=") + 1
                walltime = float(line[pos:])

        print(
            f"{problem_name}  mode: 2  it: {iteration_limit:4d}  "
            f"t: {walltime:5.1f}  f: {eval_x.f_gap:8.5f}  "
            f"g: {eval_x.g.item():8.5f}"
        )

        out["n_iterations"].append(iteration_limit)
        out["walltime"].append(walltime)
        out["f"].append(eval_x.f)
        out["f_gap"].append(eval_x.f_gap)
        out["g"].append(eval_x.g.item())
        out["x"].append(x)

        if eval_x.f_gap <= tol and eval_x.g.item() <= feas_tol:
            break

        iteration_limit += 1

    np.savez(output_file_name, **out)
    return out


def main():
    """Run the main routine of this script"""
    for problem_name in sys.argv[1:]:
        for tol in [1e-2, 1e-3]:
            print(f"solving {problem_name} with tol {tol}")
            res = run(problem_name, tol, 1e-3)
            if (
                (tol == 1e-2)
                and (res["f_gap"][-1] <= 1e-3)
                and (res["g"][-1] <= 1e-3)
            ):
                print(
                    f"skipping tol=1e-3 since tol=1e-2 was sufficient   f: {res['f_gap'][-1]}  g: {res['g'][-1]}"
                )
                break


if __name__ == "__main__":
    main()
