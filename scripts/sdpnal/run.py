# -*- coding: utf-8 -*-

"""Run SDPNAL+ on test instances"""

import os
import subprocess

import numpy as np

from cpsdppy import config as config_module
from cpsdppy import logging_helper, sdpa
from cpsdppy.sdp_solvers import common


def run(data_file_path, tol, feas_tol):
    cwd = os.getcwd()
    problem_name = os.path.splitext(os.path.basename(data_file_path))[0]

    solution_path = (
        f"{cwd}/outputs/v2/sdpnal/"
        f"{problem_name}_{tol}_{feas_tol}_tmp_sol.txt"
    )

    iteration_limit = 100

    ub = None
    lb = 1

    # Find ub
    while True:
        command = (
            f"bash scripts/sdpnal/interface.sh {data_file_path} "
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

        if eval_x.f_gap <= 2 * tol and eval_x.g.item() <= 2 * feas_tol:
            ub = iteration_limit
            break

        iteration_limit += 100

    while True:
        iteration_limit = int((lb + ub) / 2)
        command = (
            f"bash scripts/sdpnal/interface.sh {data_file_path} "
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
            f"bash scripts/sdpnal/interface.sh {data_file_path} "
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

    np.savez(f"outputs/v2/sdpnal/{problem_name}_{tol}_{feas_tol}.npz", **out)


def main():
    """Run the main routine of this script"""
    problem_names = [
        "data/SDPLIB/data/gpp250-1.dat-s",
        "data/SDPLIB/data/gpp250-2.dat-s",
        "data/SDPLIB/data/gpp250-3.dat-s",
        "data/SDPLIB/data/gpp250-4.dat-s",
        "data/SDPLIB/data/gpp500-1.dat-s",
        "data/SDPLIB/data/gpp500-2.dat-s",
        "data/SDPLIB/data/gpp500-3.dat-s",
        "data/SDPLIB/data/gpp500-4.dat-s",
        "data/SDPLIB/data/mcp250-1.dat-s",
        "data/SDPLIB/data/mcp250-2.dat-s",
        "data/SDPLIB/data/mcp250-3.dat-s",
        "data/SDPLIB/data/mcp250-4.dat-s",
        "data/SDPLIB/data/mcp500-1.dat-s",
        "data/SDPLIB/data/mcp500-2.dat-s",
        "data/SDPLIB/data/mcp500-3.dat-s",
        "data/SDPLIB/data/mcp500-4.dat-s",
        "data/rudy/out/graph_1000_5_1.dat-s",
        "data/rudy/out/graph_1000_5_2.dat-s",
        "data/rudy/out/graph_1000_5_3.dat-s",
        "data/rudy/out/graph_1000_5_4.dat-s",
        "data/rudy/out/graph_2000_5_1.dat-s",
        "data/rudy/out/graph_2000_5_2.dat-s",
        "data/rudy/out/graph_2000_5_3.dat-s",
        "data/rudy/out/graph_2000_5_4.dat-s",
        "data/rudy/out/graph_3000_5_1.dat-s",
        "data/rudy/out/graph_3000_5_2.dat-s",
        "data/rudy/out/graph_3000_5_3.dat-s",
        "data/rudy/out/graph_3000_5_4.dat-s",
        "data/rudy/out/graph_4000_5_1.dat-s",
        "data/rudy/out/graph_4000_5_2.dat-s",
        "data/rudy/out/graph_4000_5_3.dat-s",
        "data/rudy/out/graph_4000_5_4.dat-s",
        "data/rudy/out/graph_5000_5_1.dat-s",
        "data/rudy/out/graph_5000_5_2.dat-s",
        "data/rudy/out/graph_5000_5_3.dat-s",
        "data/rudy/out/graph_5000_5_4.dat-s",
        "data/rudy/out/graph_1000_10_1.dat-s",
        "data/rudy/out/graph_1000_10_2.dat-s",
        "data/rudy/out/graph_1000_10_3.dat-s",
        "data/rudy/out/graph_1000_10_4.dat-s",
        "data/rudy/out/graph_2000_10_1.dat-s",
        "data/rudy/out/graph_2000_10_2.dat-s",
        "data/rudy/out/graph_2000_10_3.dat-s",
        "data/rudy/out/graph_2000_10_4.dat-s",
        "data/rudy/out/graph_3000_10_1.dat-s",
        "data/rudy/out/graph_3000_10_2.dat-s",
        "data/rudy/out/graph_3000_10_3.dat-s",
        "data/rudy/out/graph_3000_10_4.dat-s",
        "data/rudy/out/graph_4000_10_1.dat-s",
        "data/rudy/out/graph_4000_10_2.dat-s",
        "data/rudy/out/graph_4000_10_3.dat-s",
        "data/rudy/out/graph_4000_10_4.dat-s",
        "data/rudy/out/graph_5000_10_1.dat-s",
        "data/rudy/out/graph_5000_10_2.dat-s",
        "data/rudy/out/graph_5000_10_3.dat-s",
        "data/rudy/out/graph_5000_10_4.dat-s",
        "data/rudy/out/graph_1000_15_1.dat-s",
        "data/rudy/out/graph_1000_15_2.dat-s",
        "data/rudy/out/graph_1000_15_3.dat-s",
        "data/rudy/out/graph_1000_15_4.dat-s",
        "data/rudy/out/graph_2000_15_1.dat-s",
        "data/rudy/out/graph_2000_15_2.dat-s",
        "data/rudy/out/graph_2000_15_3.dat-s",
        "data/rudy/out/graph_2000_15_4.dat-s",
        "data/rudy/out/graph_3000_15_1.dat-s",
        "data/rudy/out/graph_3000_15_2.dat-s",
        "data/rudy/out/graph_3000_15_3.dat-s",
        "data/rudy/out/graph_3000_15_4.dat-s",
        "data/rudy/out/graph_4000_15_1.dat-s",
        "data/rudy/out/graph_4000_15_2.dat-s",
        "data/rudy/out/graph_4000_15_3.dat-s",
        "data/rudy/out/graph_4000_15_4.dat-s",
        "data/rudy/out/graph_5000_15_1.dat-s",
        "data/rudy/out/graph_5000_15_2.dat-s",
        "data/rudy/out/graph_5000_15_3.dat-s",
        "data/rudy/out/graph_5000_15_4.dat-s",
        "data/rudy/out/graph_1000_20_1.dat-s",
        "data/rudy/out/graph_1000_20_2.dat-s",
        "data/rudy/out/graph_1000_20_3.dat-s",
        "data/rudy/out/graph_1000_20_4.dat-s",
        "data/rudy/out/graph_2000_20_1.dat-s",
        "data/rudy/out/graph_2000_20_2.dat-s",
        "data/rudy/out/graph_2000_20_3.dat-s",
        "data/rudy/out/graph_2000_20_4.dat-s",
        "data/rudy/out/graph_3000_20_1.dat-s",
        "data/rudy/out/graph_3000_20_2.dat-s",
        "data/rudy/out/graph_3000_20_3.dat-s",
        "data/rudy/out/graph_3000_20_4.dat-s",
        "data/rudy/out/graph_4000_20_1.dat-s",
        "data/rudy/out/graph_4000_20_2.dat-s",
        "data/rudy/out/graph_4000_20_3.dat-s",
        "data/rudy/out/graph_4000_20_4.dat-s",
        "data/rudy/out/graph_5000_20_1.dat-s",
        "data/rudy/out/graph_5000_20_2.dat-s",
        "data/rudy/out/graph_5000_20_3.dat-s",
        "data/rudy/out/graph_5000_20_4.dat-s",
    ]

    for tol in [1e-2, 1e-3]:
        for problem_name in problem_names:
            run(problem_name, tol, 1e-3)


if __name__ == "__main__":
    main()
