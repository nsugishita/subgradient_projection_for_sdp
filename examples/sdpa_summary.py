# -*- coding: utf-8 -*-

"""Script to plot summary"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    """Run the main routine of this script"""
    df = pd.read_csv("tmp/summary.csv")
    df = df[
        [
            "problem",
            "eigen",
            "n_linear_cuts",
            "n_lmi_cuts",
            "walltime",
            "n_iterations",
        ]
    ]

    df["cut_type"] = "0_nan"
    df.loc[df["n_linear_cuts"] > 0, "cut_type"] = "1_lienar"
    df.loc[df["n_lmi_cuts"] > 0, "cut_type"] = "2_lmi"
    df["n_cuts"] = df["n_linear_cuts"] + df["n_lmi_cuts"] + 0
    df = df.drop(columns=["n_linear_cuts", "n_lmi_cuts"])

    problems = df["problem"].unique()
    print(problems)

    df = df.set_index(["problem", "eigen", "cut_type", "n_cuts"])
    df = df.sort_index()
    with pd.option_context("display.max_rows", 999):
        df = df.unstack(level="eigen")
        for i in [0, 1]:
            array = df[("n_iterations", i)].values
            array[np.isnan(array)] = 0
        df["n_iterations"] = df["n_iterations"].astype(int)
        df["walltime"] = df["walltime"].round(1)

    def clip(x):
        if x > 600:
            return 600
        else:
            return x

    fig, ax = plt.subplots()
    markersize = 12
    for problem_i, problem in enumerate(problems):
        try:
            time = df.loc[(problem, "1_lienar", 1), ("walltime", 0)]
        except KeyError:
            continue
        time = clip(time)
        ax.scatter(
            problem_i,
            time,
            marker="o",
            c="None",
            edgecolors="C0",
            s=markersize,
        )

    for problem_i, problem in enumerate(problems):
        try:
            time = df.loc[(problem, "1_lienar", 1), ("walltime", 1)]
        except KeyError:
            continue
        time = clip(time)
        ax.scatter(
            problem_i, time, marker="o", c="C0", edgecolors="C0", s=markersize
        )

    for problem_i, problem in enumerate(problems):
        try:
            time = df.loc[(problem, "2_lmi", 1), ("walltime", 0)]
        except KeyError:
            continue
        time = clip(time)
        ax.scatter(
            problem_i,
            time,
            marker="o",
            c="None",
            edgecolors="C1",
            s=markersize,
        )

    for problem_i, problem in enumerate(problems):
        try:
            time = df.loc[(problem, "2_lmi", 1), ("walltime", 1)]
        except KeyError:
            continue
        time = clip(time)
        ax.scatter(
            problem_i, time, marker="o", c="C1", edgecolors="C1", s=markersize
        )

    for problem_i, problem in enumerate(problems):
        ax.axvline(problem_i, color="gray", lw=0.5, ls=":")

    ax.set_xticks(np.arange(len(problems)))
    ax.set_xticklabels(problems, rotation="vertical")
    fig.tight_layout()
    fig.savefig("tmp/summary.pdf")


if __name__ == "__main__":
    main()
