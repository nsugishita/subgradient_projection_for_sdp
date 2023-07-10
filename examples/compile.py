# -*- coding: utf-8 -*-

"""Description of this file"""

import itertools
import os
import pickle

import numpy as np
import pandas as pd

from cpsdppy import config as config_module


def main():
    """Run the main routine of this script"""
    problem_names = [
        # "gpp100",
        # "gpp124-1",
        # "gpp124-2",
        # "gpp124-3",
        # "gpp124-4",
        "gpp250-1",
        "gpp250-2",
        "gpp250-3",
        "gpp250-4",
        "gpp500-1",
        "gpp500-2",
        "gpp500-3",
        "gpp500-4",
        # "mcp100",
        # "mcp124-1",
        # "mcp124-2",
        # "mcp124-3",
        # "mcp124-4",
        "mcp250-1",
        "mcp250-2",
        "mcp250-3",
        "mcp250-4",
        "mcp500-1",
        "mcp500-2",
        "mcp500-3",
        "mcp500-4",
        "theta1",
        "theta2",
        "theta3",
        # "theta4",
        # "theta5",
        # "theta6",
    ]

    table1(problem_names)
    table2(problem_names)


def table1(problem_names):
    tols = [1e-2, 1e-3]
    methods = ["subgrad", "cosmo", "mosek"]

    iter = itertools.product(problem_names, tols, methods)

    df = pd.DataFrame(columns=["problem_name", "tol", "method", "time"])

    base_config = config_module.Config()

    cosmo_df = pd.read_csv(
        "/home/nsugishi/work/cosmo/cache/v7"
        "/compute64e2.maths.ed.ac.uk/result.csv"
    )
    cosmo_df = cosmo_df.set_index(["problem", "opt_tol"])

    for problem_name, tol, method in iter:
        config = config_module.copy(base_config)
        config.problem_name = problem_name
        config.tol = tol
        config.solver = method
        config.step_size = 1
        config.memory = 20
        config.initial_cut_type = "none"
        config.time_limit = 120
        # config.projection_after_feasibility_step = 0

        if method == "subgrad":
            config.solver = "subgradient_projection"
            config.eval_lb_every = 0
            cache_path = (
                "tmp/sdpa/v4/cache/data/"
                f"{config._asstr(only_modified=True, shorten=True)}.pkl"
            )
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    loaded = pickle.load(f)
                    time = loaded["walltime"]
            else:
                time = np.nan

        elif method == "cosmo":
            time = cosmo_df.loc[(problem_name, tol), "time"]
        elif method == "mosek":
            time = np.nan
        else:
            raise ValueError(f"unknown method name '{method}'")

        df.loc[len(df)] = (problem_name, tol, method, time)
    df["time"] = df["time"].astype(float)
    df["time"] = df["time"].clip(None, 120)

    with pd.option_context("display.max_rows", 999):
        _df = df.set_index(["problem_name", "tol", "method"]).unstack(
            level=["tol", "method"]
        )
        _df = _df.loc[problem_names]
        print(_df.round(1))
        # print(_df.round(1).astype(str).to_latex().replace("nan", "-"))


def table2(problem_names):
    tols = [1e-2, 1e-3]

    cut_types = ["single", "comb", "single+comb"]

    iter = itertools.product(problem_names, tols, cut_types)

    df = pd.DataFrame(columns=["problem_name", "tol", "cut_type", "time"])

    base_config = config_module.Config()

    cosmo_df = pd.read_csv(
        "/home/nsugishi/work/cosmo/cache/v7"
        "/compute64e2.maths.ed.ac.uk/result.csv"
    )
    cosmo_df = cosmo_df.set_index(["problem", "opt_tol"])

    for problem_name, tol, cut_type in iter:
        config = config_module.copy(base_config)
        config.problem_name = problem_name
        config.tol = tol
        config.solver = "subgradient_projection"
        config.step_size = 1
        config.memory = 20
        config.initial_cut_type = "none"
        config.time_limit = 120
        config.eval_lb_every = 0

        if cut_type == "none":
            config.eigen_comb_cut = 0
            config.n_linear_cuts = 0
        elif cut_type == "single":
            config.eigen_comb_cut = 0
            config.n_linear_cuts = 1
        elif cut_type == "comb":
            config.eigen_comb_cut = 1
            config.n_linear_cuts = 0
        elif cut_type == "single+comb":
            config.eigen_comb_cut = 1
            config.n_linear_cuts = 1
        else:
            raise ValueError
        cache_path = (
            "tmp/sdpa/v4/cache/data/"
            f"{config._asstr(only_modified=True, shorten=True)}.pkl"
        )
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                loaded = pickle.load(f)
                time = loaded["walltime"]
        else:
            time = np.nan

        df.loc[len(df)] = (problem_name, tol, cut_type, time)
    df["time"] = df["time"].astype(float)
    df["time"] = df["time"].clip(None, 120)

    with pd.option_context("display.max_rows", 999):
        _df = df.set_index(["problem_name", "tol", "cut_type"]).unstack(
            level=["tol", "cut_type"]
        )
        # idx = pd.IndexSlice
        # _df = _df.loc[idx[problem_names], idx["time", tols, cut_types]]
        print(_df.round(1))
        # print(_df.round(1).astype(str).to_latex().replace("nan", "-"))


if __name__ == "__main__":
    main()

# vimquickrun: python %