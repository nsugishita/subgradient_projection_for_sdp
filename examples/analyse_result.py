# -*- coding: utf-8 -*-

"""Description of this file"""

import pickle

import numpy as np
import pandas as pd

# 0: The original
# 1: Ignore infeasibility in Mosek
mode = 0

# 'walltime' or 'n_iterations'
item = "n_iterations"

latex = True

def load_subgradient_projection_result(problem_name, tol):
    if mode == 2:
        if tol != 1e-3:
            path = (
                "outputs/sdplib/v2/result/"
                f"problem_{problem_name}_solver_subgradient_projection_"
                f"tol_{tol}_feas_{tol}_n_linear_cuts_0.pkl"
            )
        else:
            path = (
                "outputs/sdplib/v2/result/"
                f"problem_{problem_name}_solver_subgradient_projection_"
                f"tol_{tol}_n_linear_cuts_0.pkl"
            )
    else:
        path = (
            "outputs/sdplib/v2/result/"
            f"problem_{problem_name}_solver_subgradient_projection_tol_{tol}_"
            "n_linear_cuts_0.pkl"
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    if item == "walltime":
        return data["walltime"]
    elif item == "n_iterations":
        return data["n_iterations"]
    else:
        raise ValueError(f"unknown item: {item}")


def load_mosek_result(problem_name, tol):
    if mode >= 1:
        df = pd.read_csv("outputs/revised/mosek/walltime.csv")
        return df[(df["problem_name"] == problem_name) & (df["tol"] == tol)][item].item()
    else:
        path = (
            "outputs/sdplib/v2/result/"
            f"problem_{problem_name}_solver_mosek_tol_{tol}_n_linear_cuts_0.pkl"
        )
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data[item]


def load_cosmo_result(problem_name, tol):
    data = np.load(
        f"cosmo_results/run_cosmo_with_iteration_limit/{problem_name}_{tol}.npz"
    )
    if item == "walltime":
        return data["time"]
    elif item == "n_iterations":
        return data["iter"]
    else:
        raise ValueError(f"unknown item: {item}")


def load_result(problem_name, solver_name, tol):
    if solver_name == "subgrad":
        return load_subgradient_projection_result(problem_name, tol)
    elif solver_name == "mosek":
        return load_mosek_result(problem_name, tol)
    elif solver_name == "cosmo":
        return load_cosmo_result(problem_name, tol)
    else:
        raise ValueError(f"unknown solver: {solver_name}")


def main():
    """Run the main routine of this script"""
    problem_names = [
        "gpp250-1",
        "gpp250-2",
        "gpp250-3",
        "gpp250-4",
        "gpp500-1",
        "gpp500-2",
        "gpp500-3",
        "gpp500-4",
        "mcp250-1",
        "mcp250-2",
        "mcp250-3",
        "mcp250-4",
        "mcp500-1",
        "mcp500-2",
        "mcp500-3",
        "mcp500-4",
    ]
    first = True
    print(f"{'':>8s}  ", end="")
    first = False
    for tol in [1e-2, 1e-3]:
        for solver_i, solver_name in enumerate(["subgrad", "mosek", "cosmo"]):
            if latex and not first:
                print("& ", end="")
            if solver_i == 0:
                print(f"tol: {tol * 100:3.1f} ", end="")
            else:
                print(f"{'':>8s} ", end="")
            first = False
    if latex:
        print("\\\\")
    else:
        print()
    first = True
    print(f"{'problem':>8s}  ", end="")
    first = False
    for tol in [1e-2, 1e-3]:
        for solver_name in ["subgrad", "mosek", "cosmo"]:
            if latex and not first:
                print("& ", end="")
            print(f"{solver_name:>8s} ", end="")
            first = False
    if latex:
        print("\\\\")
    else:
        print()
    for problem_name in problem_names:
        first = True
        print(f"{problem_name:>8s}  ", end="")
        first = False
        for tol in [1e-2, 1e-3]:
            for solver_name in ["subgrad", "mosek", "cosmo"]:
                if latex and not first:
                    print("& ", end="")
                v = load_result(problem_name, solver_name, tol)
                print(f"{v:8.1f} ", end="")
                first = False
        if latex:
            print("\\\\")
        else:
            print()


if __name__ == "__main__":
    main()
