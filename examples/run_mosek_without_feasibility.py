# -*- coding: utf-8 -*-

"""Run solvers with various configurations

This runs the subgradient projection solver and MOSEK with various
configurations and gathers the result.
"""

import numpy as np
import pandas as pd
import collections
from cpsdppy.sdp_solvers import common
from cpsdppy import config as config_module
from cpsdppy import logging_helper, sdpa
from cpsdppy.sdp_solvers import mosek


def main():
    df = pd.DataFrame(columns=["problem_name", "tol", "walltime"])
    problem_names = [
        "mcp100",
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

    for problem_name in problem_names:
        config = config_module.Config()
        config.log_to_stdout = 1
        config.problem_name = problem_name
        config.solver = "mosek"
        problem_data = sdpa.read(config)

        n_iterations_to_tol = collections.defaultdict(list)
        n_iterations_to_res = {}

        tol = 1e-4
        config.tol = tol
        config.feas_tol = tol
        config.log_to_stdout = 1
        res = mosek.run(problem_data, config)
        gap = (
            res["iter_dual_objective"]
            - problem_data["target_objective"]
        ) / np.abs(problem_data["target_objective"])
        for tol in [1e-2, 1e-3]:
            walltime = np.min(res["iter_walltime"][gap <= tol])
            df.loc[len(df)] = (config.problem_name, tol, walltime)

    df.to_csv("outputs/revised/mosek/walltime.csv")


if __name__ == "__main__":
    main()
