# -*- coding: utf-8 -*-

"""Solve SDPA

This scripts run the subgradient projection to solve a test instance in SDPLIB.
"""

import copy
import argparse
import logging
import os
import pickle
import typing

from cpsdppy import config as config_module
from cpsdppy import logging_helper, sdpa
from cpsdppy.sdp_solvers import cosmo, mosek, subgradient_projection

logger = logging.getLogger(__name__)


def run(input_file_path, config, result_file_path):
    """Run the solver on the current process"""
    if os.path.exists(result_file_path):
        return

    os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
    log_file_path = os.path.splitext(result_file_path)[0] + ".txt"

    logger.info("result are saved in:")
    logger.info(result_file_path)
    logger.info("log messages are saved in:")
    logger.info(log_file_path)

    problem_data = sdpa.read(input_file_path)

    with logging_helper.save_log(log_file_path):
        if config.solver == "subgradient_projection":
            res = subgradient_projection.run(problem_data, config)
        elif config.solver == "mosek":
            res = mosek.run(problem_data, config)
        elif config.solver == "":
            raise ValueError("config.solver is missing")
        else:
            raise ValueError(f"unknown solver '{config.solver}'")

    with open(result_file_path, "wb") as f:
        pickle.dump(res, f)


input_file_paths1 = [
    "data/SDPLIB/data/mcp250-1.dat-s",
    "data/SDPLIB/data/mcp250-2.dat-s",
    "data/SDPLIB/data/mcp250-3.dat-s",
    "data/SDPLIB/data/mcp250-4.dat-s",
    "data/SDPLIB/data/mcp500-1.dat-s",
    "data/SDPLIB/data/mcp500-2.dat-s",
    "data/SDPLIB/data/mcp500-3.dat-s",
    "data/SDPLIB/data/mcp500-4.dat-s",
    "data/SDPLIB/data/gpp250-1.dat-s",
    "data/SDPLIB/data/gpp250-2.dat-s",
    "data/SDPLIB/data/gpp250-3.dat-s",
    "data/SDPLIB/data/gpp250-4.dat-s",
    "data/SDPLIB/data/gpp500-1.dat-s",
    "data/SDPLIB/data/gpp500-2.dat-s",
    "data/SDPLIB/data/gpp500-3.dat-s",
    "data/SDPLIB/data/gpp500-4.dat-s",
]

input_file_paths2 = [
    "data/rudy/out/weighted_graph_1000_5_1.dat-s",
    "data/rudy/out/weighted_graph_1000_5_2.dat-s",
    "data/rudy/out/weighted_graph_1000_5_3.dat-s",
    "data/rudy/out/weighted_graph_1000_5_4.dat-s",
    "data/rudy/out/weighted_graph_2000_5_1.dat-s",
    "data/rudy/out/weighted_graph_2000_5_2.dat-s",
    "data/rudy/out/weighted_graph_2000_5_3.dat-s",
    "data/rudy/out/weighted_graph_2000_5_4.dat-s",
    "data/rudy/out/weighted_graph_3000_5_1.dat-s",
    "data/rudy/out/weighted_graph_3000_5_2.dat-s",
    "data/rudy/out/weighted_graph_3000_5_3.dat-s",
    "data/rudy/out/weighted_graph_3000_5_4.dat-s",
    "data/rudy/out/weighted_graph_4000_5_1.dat-s",
    "data/rudy/out/weighted_graph_4000_5_2.dat-s",
    "data/rudy/out/weighted_graph_4000_5_3.dat-s",
    "data/rudy/out/weighted_graph_4000_5_4.dat-s",
    "data/rudy/out/weighted_graph_5000_5_1.dat-s",
    "data/rudy/out/weighted_graph_5000_5_2.dat-s",
    "data/rudy/out/weighted_graph_5000_5_3.dat-s",
    "data/rudy/out/weighted_graph_5000_5_4.dat-s",
    "data/rudy/out/weighted_graph_1000_10_1.dat-s",
    "data/rudy/out/weighted_graph_1000_10_2.dat-s",
    "data/rudy/out/weighted_graph_1000_10_3.dat-s",
    "data/rudy/out/weighted_graph_1000_10_4.dat-s",
    "data/rudy/out/weighted_graph_2000_10_1.dat-s",
    "data/rudy/out/weighted_graph_2000_10_2.dat-s",
    "data/rudy/out/weighted_graph_2000_10_3.dat-s",
    "data/rudy/out/weighted_graph_2000_10_4.dat-s",
    "data/rudy/out/weighted_graph_3000_10_1.dat-s",
    "data/rudy/out/weighted_graph_3000_10_2.dat-s",
    "data/rudy/out/weighted_graph_3000_10_3.dat-s",
    "data/rudy/out/weighted_graph_3000_10_4.dat-s",
    "data/rudy/out/weighted_graph_4000_10_1.dat-s",
    "data/rudy/out/weighted_graph_4000_10_2.dat-s",
    "data/rudy/out/weighted_graph_4000_10_3.dat-s",
    "data/rudy/out/weighted_graph_4000_10_4.dat-s",
    "data/rudy/out/weighted_graph_5000_10_1.dat-s",
    "data/rudy/out/weighted_graph_5000_10_2.dat-s",
    "data/rudy/out/weighted_graph_5000_10_3.dat-s",
    "data/rudy/out/weighted_graph_5000_10_4.dat-s",
    "data/rudy/out/weighted_graph_1000_15_1.dat-s",
    "data/rudy/out/weighted_graph_1000_15_2.dat-s",
    "data/rudy/out/weighted_graph_1000_15_3.dat-s",
    "data/rudy/out/weighted_graph_1000_15_4.dat-s",
    "data/rudy/out/weighted_graph_2000_15_1.dat-s",
    "data/rudy/out/weighted_graph_2000_15_2.dat-s",
    "data/rudy/out/weighted_graph_2000_15_3.dat-s",
    "data/rudy/out/weighted_graph_2000_15_4.dat-s",
    "data/rudy/out/weighted_graph_3000_15_1.dat-s",
    "data/rudy/out/weighted_graph_3000_15_2.dat-s",
    "data/rudy/out/weighted_graph_3000_15_3.dat-s",
    "data/rudy/out/weighted_graph_3000_15_4.dat-s",
    "data/rudy/out/weighted_graph_4000_15_1.dat-s",
    "data/rudy/out/weighted_graph_4000_15_2.dat-s",
    "data/rudy/out/weighted_graph_4000_15_3.dat-s",
    "data/rudy/out/weighted_graph_4000_15_4.dat-s",
    "data/rudy/out/weighted_graph_5000_15_1.dat-s",
    "data/rudy/out/weighted_graph_5000_15_2.dat-s",
    "data/rudy/out/weighted_graph_5000_15_3.dat-s",
    "data/rudy/out/weighted_graph_5000_15_4.dat-s",
    "data/rudy/out/weighted_graph_1000_20_1.dat-s",
    "data/rudy/out/weighted_graph_1000_20_2.dat-s",
    "data/rudy/out/weighted_graph_1000_20_3.dat-s",
    "data/rudy/out/weighted_graph_1000_20_4.dat-s",
    "data/rudy/out/weighted_graph_2000_20_1.dat-s",
    "data/rudy/out/weighted_graph_2000_20_2.dat-s",
    "data/rudy/out/weighted_graph_2000_20_3.dat-s",
    "data/rudy/out/weighted_graph_2000_20_4.dat-s",
    "data/rudy/out/weighted_graph_3000_20_1.dat-s",
    "data/rudy/out/weighted_graph_3000_20_2.dat-s",
    "data/rudy/out/weighted_graph_3000_20_3.dat-s",
    "data/rudy/out/weighted_graph_3000_20_4.dat-s",
    "data/rudy/out/weighted_graph_4000_20_1.dat-s",
    "data/rudy/out/weighted_graph_4000_20_2.dat-s",
    "data/rudy/out/weighted_graph_4000_20_3.dat-s",
    "data/rudy/out/weighted_graph_4000_20_4.dat-s",
    "data/rudy/out/weighted_graph_5000_20_1.dat-s",
    "data/rudy/out/weighted_graph_5000_20_2.dat-s",
    "data/rudy/out/weighted_graph_5000_20_3.dat-s",
    "data/rudy/out/weighted_graph_5000_20_4.dat-s",
];



def run_all():
    for input_file_path in input_file_paths1:
        for tol in [1e-2, 1e-3]:
            for eigen_comb_cut in [0, 1]:
                for n_linear_cuts in [0, 1]:
                    if n_linear_cuts == 0 and eigen_comb_cut == 0:
                        continue
                    problem_name = os.path.splitext(os.path.basename(input_file_path))[0]
                    config = config_module.Config()
                    config.tol = tol
                    config.solver = "subgradient_projection"
                    config.eigen_comb_cut = eigen_comb_cut
                    config.n_linear_cuts = n_linear_cuts
                    result_file_path = (
                        f"outputs/v2/{config.solver}/{problem_name}_"
                        f"tol_{config.tol}_comb_{config.eigen_comb_cut}_"
                        f"linear_{config.n_linear_cuts}.pkl"
                    )
                    run(input_file_path, config, result_file_path)

    for input_file_path in input_file_paths1 + input_file_paths2:
        for tol in [1e-2, 1e-3]:
            for solver in ["subgradient_projection", "mosek"]:
                eigen_comb_cut = 1
                n_linear_cuts = 0
                problem_name = os.path.splitext(os.path.basename(input_file_path))[0]
                config = config_module.Config()
                config.tol = tol
                config.solver = solver
                config.eigen_comb_cut = eigen_comb_cut
                config.n_linear_cuts = n_linear_cuts
                result_file_path = (
                    f"outputs/v2/{config.solver}/{problem_name}_"
                    f"tol_{config.tol}_comb_{config.eigen_comb_cut}_"
                    f"linear_{config.n_linear_cuts}.pkl"
                )
                run(input_file_path, config, result_file_path)

def main():
    """Run the entry point of this script"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tol",
        type=float,
        choices=[1e-2, 1e-3],
        required=True,
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["subgradient_projection", "mosek"],
        required=True,
    )
    parser.add_argument(
        "--comb",
        type=int,
        choices=[0, 1],
    )
    parser.add_argument(
        "--linear",
        type=int,
        choices=[0, 1],
    )
    args = parser.parse_args()

    if args.comb == 0 and args.linear == 0:
        return

    logging_helper.setup(dir=None)

    config = config_module.Config()
    config.problem_name = args.problem
    config.tol = args.tol
    config.solver = args.solver
    if args.comb is not None:
        config.eigen_comb_cut = args.comb
    if args.linear is not None:
        config.n_linear_cuts = args.linear

    problem_name = os.path.splitext(os.path.basename(config.problem_name))[0]

    result_file_path = (
        f"outputs/v2/{config.solver}/{problem_name}_"
        f"tol_{config.tol}_comb_{config.eigen_comb_cut}_"
        f"linear_{config.n_linear_cuts}.pkl"
    )
    run(config.problem_name, config, result_file_path)

if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
