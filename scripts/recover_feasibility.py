# -*- coding: utf-8 -*-

"""Recover feasibility of an output of subgradient projection method"""

import copy
import numpy as np
import argparse
import logging
import os
import pickle
import typing

from cpsdppy import config as config_module
from cpsdppy import logging_helper, sdpa
from cpsdppy.sdp_solvers import mosek, subgradient_projection

logger = logging.getLogger(__name__)


def run(input_file_path, config, loaded_file_path, output_file_path):
    """Run the solver on the current process"""
    if not os.path.exists(loaded_file_path):
        print(f"file not found: {loaded_file_path}")
        return
    if os.path.exists(output_file_path):
        print(f"file already exists: {output_file_path}")
        return

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    log_file_path = os.path.splitext(output_file_path)[0] + ".txt"

    problem_data = sdpa.read(input_file_path)

    with open(loaded_file_path, "rb") as f:
        res = pickle.load(f)

    if res["iter_fv_gap"][-1] <= config.tol and res["iter_gv"][-1] <= 1e-3:
        problem_data["initial_x"] = res["iter_v"][-1]
    else:
        problem_data["initial_x"] = res["iter_x"][-2]
    config.feasibility_recovery = 1
    config.tol = 0
    config.feas_tol = 1e-6

    with logging_helper.save_log(log_file_path):
        res2 = subgradient_projection.run(problem_data, config)

    with open(output_file_path, "wb") as f:
        pickle.dump(res2, f)


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
        default=1,
    )
    parser.add_argument(
        "--linear",
        type=int,
        choices=[0, 1],
        default=0,
    )
    args = parser.parse_args()

    if args.comb == 0 and args.linear == 0:
        return

    _impl(args)


def _impl(args):
    logging_helper.setup(dir=None)

    config = config_module.Config()
    config.problem_name = args.problem
    config.tol = args.tol
    config.solver = args.solver
    config.time_limit = 5 * 60 * 60
    if args.comb is not None:
        config.eigen_comb_cut = args.comb
    if args.linear is not None:
        config.n_linear_cuts = args.linear

    problem_name = os.path.splitext(os.path.basename(config.problem_name))[0]

    input_file_path = (
        f"outputs/v2/{config.solver}/{problem_name}_"
        f"tol_{config.tol}_comb_{config.eigen_comb_cut}_"
        f"linear_{config.n_linear_cuts}.pkl"
    )
    output_file_path = (
        f"outputs/vfeas/{config.solver}/{problem_name}_"
        f"tol_{config.tol}_comb_{config.eigen_comb_cut}_"
        f"linear_{config.n_linear_cuts}_feasibility_recovery.pkl"
    )
    run(config.problem_name, config, input_file_path, output_file_path)

if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
