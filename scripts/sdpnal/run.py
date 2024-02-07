# -*- coding: utf-8 -*-

"""Description of this file"""

import sys
import numpy as np
from cpsdppy import config as config_module
from cpsdppy import logging_helper, sdpa
import subprocess
import os
from cpsdppy.sdp_solvers import common


def run(problem_name, tol, feas_tol):
    cwd = os.getcwd()

    solution_path = (
        f"{cwd}/outputs/revised/sdpnal/"
        f"{problem_name}_{tol}_{feas_tol}_tmp_sol.txt"
    )

    iteration_limit = 100

    ub = None
    lb = 1

    # Find ub
    while True:
        command = (
            f"bash scripts/sdpnal/interface.sh {problem_name} "
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
        config.problem_name = problem_name
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

        if eval_x.f_gap <= 2 * tol and eval_x.g.item() <= 2 * feas_tol:
            ub = iteration_limit
            break

        iteration_limit += 100

    while True:
        iteration_limit = int((lb + ub) / 2)
        command = (
            f"bash scripts/sdpnal/interface.sh {problem_name} "
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
        config.problem_name = problem_name
        problem_data = sdpa.read(config)

        eval_x = common.evaluate_solution(x, problem_data)
        lines = res.stdout.split("\n")
        for line in lines:
            if "Computing time (total)" in line:
                pos = line.index("=") + 1
                walltime = float(line[pos:])

        if eval_x.f_gap <= 2 * tol and eval_x.g.item() <= 2 * feas_tol:
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

        if ub - lb <= 10:
            break

    iteration_limit = lb

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
            f"bash scripts/sdpnal/interface.sh {problem_name} "
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
        config.problem_name = problem_name
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

    np.savez(
        f"outputs/revised/sdpnal/{problem_name}_{tol}_{feas_tol}.npz", **out
    )


def main():
    """Run the main routine of this script"""
    if len(sys.argv) != 2 or sys.argv[1] not in ["1", "2"]:
        print(f"usage: python {sys.argv[0]} 1|2")

    if sys.argv[1] == "1":
        problem_names = [
            "mcp250-1",
            "mcp250-2",
            "mcp250-3",
            "mcp250-4",
            "mcp500-1",
            "mcp500-2",
            "mcp500-3",
            "mcp500-4",
        ]
    elif sys.argv[1] == "2":
        problem_names = [
            "gpp250-1",
            "gpp250-2",
            "gpp250-3",
            "gpp250-4",
            "gpp500-1",
            "gpp500-2",
            "gpp500-3",
            "gpp500-4",
        ]

    for tol in [1e-2, 1e-3]:
        for problem_name in problem_names:
            run(problem_name, tol, 1e-3)


if __name__ == "__main__":
    main()
