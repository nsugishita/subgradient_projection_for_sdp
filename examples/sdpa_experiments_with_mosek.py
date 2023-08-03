# -*- coding: utf-8 -*-

"""Solve SDPA using column generation"""

import argparse
import logging
import os
import subprocess

import pandas as pd

from cpsdppy import logging_helper

logger = logging.getLogger(__name__)


version = "vdev"
result_dir = f"tmp/sdpa/{version}/result"
returncode_file = f"{result_dir}/returncode.csv"


def main() -> None:
    """Run the main routine of this script"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem-name",
        type=str,
        nargs="+",
        default=[
            "theta1",
            "theta2",
            # "theta3",
            # "gpp100",
            # "gpp124-1",
            # "gpp124-2",
            # "gpp124-3",
            # "gpp124-4",
            "gpp500-1",
            "gpp500-2",
            "gpp500-3",
            "gpp500-4",
            "gpp250-1",
            "gpp250-2",
            "gpp250-3",
            "gpp250-4",
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
        ],
    )
    parser.add_argument(
        "--solver",
        type=str,
        nargs="+",
        default=[
            "subgradient_projection",
            "mosek",
        ],
    )
    parser.add_argument(
        "--tol",
        type=float,
        nargs="+",
        default=[
            1e-2,
            1e-3,
        ],
    )
    args = parser.parse_args()

    logging_helper.setup()

    if os.path.exists(returncode_file):
        returncode_df = pd.read_csv(returncode_file)
    else:
        returncode_df = pd.DataFrame(
            columns=["problem", "solver", "tol", "returncode"]
        )

    returncode_df = returncode_df.set_index(["problem", "solver", "tol"])

    for problem_name in args.problem_name:
        for solver in args.solver:
            for tol in args.tol:
                command = (
                    "python examples/solve_sdpa.py "
                    f"--dir {result_dir} "
                    f"--problem-name {problem_name} "
                    f"--solver {solver} "
                    f"--tol {tol}"
                )
                logger.info("- " * 30)
                logger.info(f"problem: {problem_name}  tol: {tol:.1e}")
                logger.info(f"command: {command}")
                logger.info("- " * 30)

                try:
                    returncode = returncode_df.loc[
                        (problem_name, solver, tol), "returncode"
                    ]
                    cached = True
                except KeyError:
                    cached = False

                if not cached:
                    ret = subprocess.run(command, shell=True)
                    returncode = ret.returncode

                logger.info("= " * 30)
                logger.info(
                    f"problem: {problem_name}  tol: {tol:.1e}  "
                    f"cached: {int(cached)}  "
                    f"returncode: {returncode}"
                )
                logger.info("= " * 30)

                if not cached:
                    returncode_df.loc[
                        (problem_name, solver, tol), :
                    ] = returncode
                    returncode_df.to_csv(returncode_file)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()

# vimquickrun: . ./scripts/activate.sh ; python %
