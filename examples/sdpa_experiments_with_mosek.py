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
result_dir = f"tmp/sdpa/{version}/cache/result"
returncode_file = f"{result_dir}/returncode.csv"


def main() -> None:
    """Run the main routine of this script"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem-names",
        type=str,
        nargs="+",
        default=[
            "theta1",
            "theta2",
            "theta3",
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
        df = pd.read_csv(returncode_file)
    else:
        df = pd.DataFrame(columns=["problem", "tol", "returncode"])

    df = df.set_index(["problem", "tol"])

    for problem_name in args.problem_names:
        for tol in args.tol:
            command = (
                "python examples/solve_sdpa_with_mosek.py "
                f"--dir {result_dir} "
                f"--problem-name {problem_name} --tol {tol}"
            )
            logger.info("- " * 30)
            logger.info(f"problem: {problem_name}  tol: {tol:.1e}")
            logger.info(f"command: {command}")
            logger.info("- " * 30)

            ret = subprocess.run(command, shell=True)

            logger.info("= " * 30)
            logger.info(
                f"problem: {problem_name}  tol: {tol:.1e}  "
                f"returncode: {ret.returncode}"
            )
            logger.info("= " * 30)

            df.loc[(problem_name, tol), :] = ret.returncode

            df.to_csv(returncode_file)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()

# vimquickrun: . ./scripts/activate.sh ; python %
