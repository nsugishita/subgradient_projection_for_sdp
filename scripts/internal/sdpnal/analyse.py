# -*- coding: utf-8 -*-

"""Description of this file"""

import numpy as np


def main():
    """Run the main routine of this script"""
    problem_names = [
        "mcp250-1",
        "mcp250-2",
        "mcp250-3",
        "mcp250-4",
        "mcp500-1",
        "mcp500-2",
        "mcp500-3",
        "mcp500-4",
        "gpp250-1",
        "gpp250-2",
        "gpp250-3",
        "gpp250-4",
        "gpp500-1",
        "gpp500-2",
        "gpp500-3",
        "gpp500-4",
    ]
    tols = [0.01, 0.001]
    dir_path = "outputs/v2/sdpnal/"
    print(f"{'tol':>8s}  ", end="")
    for tol_i, tol in enumerate(tols):
        print(f"{tol:6.3f}", end="  ")
        print(f"{'':>4s}", end="  ")
        if tol_i < len(tols) - 1:
            print("  |  ", end="")
    print()
    print(f"{'problem':>8s}  ", end="")
    for tol_i, tol in enumerate(tols):
        print(f"{'time':>6s}", end="  ")
        print(f"{'it':>4s}", end="  ")
        if tol_i < len(tols) - 1:
            print("  |  ", end="")
    print()
    for problem_name in problem_names:
        print(f"{problem_name:>8s}  ", end="")
        for tol_i, tol in enumerate(tols):
            file_name = f"{problem_name}_{tol}_0.001.npz"
            d = np.load(dir_path + file_name)
            print(f"{d['walltime'][-1]:6.1f}", end="  ")
            print(f"{d['n_iterations'][-1]:4d}", end="  ")
            if tol_i < len(tols) - 1:
                print("  |  ", end="")
        print()


if __name__ == "__main__":
    main()
