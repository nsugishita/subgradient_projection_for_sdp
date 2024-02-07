# -*- coding: utf-8 -*-

"""Description of this file"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

dir_name = "examples/rudy/res/"

def main():
    """Run the main routine of this script"""
    sizes = [1000, 2000, 3000, 4000, 5000]
    densities = [5, 10]
    random_seeds = [1, 2]
    solvers = ["subgradient_projection", "mosek"]
    # tols = [0.01, 0.001]
    tols = [0.01]

    res = {}

    for density in densities:
        for size in sizes:
            for random_seed in random_seeds:
                for solver in solvers:
                    for tol in tols:
                        path = (
                            f"size_{size}_density_{density}_"
                            f"random_seed_{random_seed}_solver_{solver}_tol_{tol}_"
                            "n_linear_cuts_0.pkl"
                        )
                        with open(f"{dir_name}/{path}", "rb") as f:
                            res[density, size, random_seed, solver, tol] = pickle.load(f)


    fig, axes = plt.subplots(len(random_seeds) * len(densities), 1, sharex=True, sharey=True)
    for density_i, density in enumerate(densities):
        for random_seed_i, random_seed in enumerate(random_seeds):
            ax = axes[density_i * len(random_seeds) + random_seed_i]
            for tol_i, tol in enumerate(tols):
                for solver_i, solver in enumerate(solvers):
                    walltime = [res[density, size, random_seed, solver, tol]["walltime"] for size in sizes]
                    ls = "-" if tol_i == 0 else "--"
                    color = f"C{solver_i}"
                    label = solver if density_i == 0 and random_seed_i == 0 else None
                    ax.plot(sizes, walltime, ls=ls, color=color, marker="o", markersize=4, label=label)
                    ax.text(0.6, 0.94, f'density {density}  instance {random_seed}', ha="center", va="top", transform=ax.transAxes)
    axes[-1].set_xticks(sizes)
    axes[-1].set_xlabel("size")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(f"{dir_name}/walltime_vs_size.png")

    print(f"{'  ':2s}  {' '}  ", end="")
    for size_i, size in enumerate(sizes):
        for solver_i, solver in enumerate(solvers):
            if solver_i == 0:
                print(f"{size:7d}  ", end="")
            else:
                print(f"{' ':7s}  ", end="")
    print()
    print(f"{'de':2s}  {' '}  ", end="")
    for size_i, size in enumerate(sizes):
        for solver_i, solver in enumerate(solvers):
            print(f"{solver[:7]:>7s}  ", end="")
    print()
    for density_i, density in enumerate(densities):
        for random_seed_i, random_seed in enumerate(random_seeds):
            print(f"{density:2d}  {random_seed}  ", end="")
            ax = axes[density_i * len(random_seeds) + random_seed_i]
            # for tol_i, tol in enumerate(tols):
            for size_i, size in enumerate(sizes):
                for solver_i, solver in enumerate(solvers):
                    walltime = res[density, size, random_seed, solver, tol]["primal_objective"]
                    print(f"{walltime:7.1f}  ", end="")
            print()

if __name__ == "__main__":
    main()
