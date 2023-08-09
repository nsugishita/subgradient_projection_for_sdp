# -*- coding: utf-8 -*-

"""Run solvers with various configurations

This runs the subgradient projection solver and MOSEK with various
configurations and gathers the result.
"""

import argparse
import collections
import itertools
import logging

import numpy as np
import pandas as pd

from cpsdppy import config as config_module
from cpsdppy import logging_helper
from examples import solve_sdpa

logger = logging.getLogger(__name__)

# v9: simpler step size manager

version = "v9"
result_dir = f"tmp/sdpa/{version}/result"


def main() -> None:
    """Run the main routine of this script"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem-names",
        type=str,
        nargs="+",
        default=[
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
            # "theta1",
            # "theta2",
            # "theta3",
            # "theta4",
            # "theta5",
            # "theta6",
        ],
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )
    config_module.add_arguments(parser)
    args = parser.parse_args()

    base_config = config_module.Config()
    config_module.parse_args(base_config, args)

    logging_helper.setup()

    if args.smoke_test:
        setups = list(
            namedtuples_from_product(
                "setup",
                "problem_name",
                args.problem_names,
                "tol",
                [1e-3],
                "n_linear_cuts",
                [0],
                "eigen_comb_cut",
                [1],
            )
        )
        _impl(base_config, setups)

    else:
        setups = list(
            namedtuples_from_product(
                "setup",
                "problem_name",
                args.problem_names,
                "solver",
                ["subgradient_projection"],
                "tol",
                [1e-2, 1e-3],
                "n_linear_cuts",
                [0, 1],
                "eigen_comb_cut",
                [0, 1],
            )
        )
        _impl(base_config, setups)

        setups = list(
            namedtuples_from_product(
                "setup",
                "problem_name",
                args.problem_names,
                "solver",
                ["cosmo", "mosek", "subgradient_projection"],
                "tol",
                [1e-2, 1e-3],
                "n_linear_cuts",
                [0],
                "eigen_comb_cut",
                [1],
            )
        )
        _impl(base_config, setups)


def _impl(base_config, setups):
    def config_filter(config):
        if config.n_linear_cuts == 0:
            if config.eigen_comb_cut == 0:
                return False
        return True

    run_data: list = []
    for setup in setups:
        config = base_config._update_from_dict(setup._asdict())
        if not config_filter(config):
            continue

        logger.info("- " * 20)
        logger.info(str(setup))
        logger.info("- " * 20)

        returncode, result = solve_sdpa.run_subprocess(config, result_dir)
        run_data.append((config._astuple(shorten=True), returncode, result))

        summary(run_data)


def summary(run_data):
    """Print summary

    This takes a list of 2-tuple, `(configurations, returncode, results)`.
    `configurations` is a tuple of pairs `(configuration, value)`,
    while `resuls` is a dictionary with `walltime` and `n_iterations`.
    """
    if len(run_data) <= 1:
        return
    records = []
    for config_tuple, returncode, result in run_data:
        if result is not None:
            walltime = result["walltime"]
            n_iterations = result["n_iterations"]
        else:
            walltime = np.nan
            n_iterations = np.nan
        records.append(
            tuple(
                config_tuple
                + (
                    ("returncode", returncode),
                    ("walltime", walltime),
                    ("n_iterations", n_iterations),
                )
            )
        )
    df = pd.DataFrame.from_records([{k: v for k, v in x} for x in records])
    dropped = []
    for column in list(df.columns[:-2]):
        if np.unique(df[column]).size == 1:
            dropped.append(column)
    df.drop(columns=dropped, inplace=True)
    for tpls, _, _ in run_data:
        index = [k for k, _ in tpls if k not in dropped]
        break
    try:
        tol_position = index.index("tol")
    except ValueError:
        tol_position = -1
    if "solver" in df:
        df["solver"][df["solver"] == "subgradient_projection"] = "subgrad"
    df = df.set_index(index)
    df = df.sort_index()
    df["walltime"] = df["walltime"].astype(float)
    df["walltime"] = np.round(df["walltime"].values, 2)
    with pd.option_context("display.max_rows", 999):
        if tol_position >= 0:
            print(df.unstack(level=tol_position))
        else:
            print(df)
        with open(f"{result_dir}/summary.txt", "w") as f:
            f.write(df.to_string())
            f.write("\n")
    df.to_csv(f"{result_dir}/summary.csv")


def namedtuples_from_product(name, *args):
    """Create a sequence of namedtuples from the products of given items

    Examples
    --------
    >>> a = namedtuples_from_product(
    ...     "items", "name", ["foo", "bar"], "age", [10, 20, 30])
    >>> for x in a:
    ...     print(x)
    items(name='foo', age=10)
    items(name='foo', age=20)
    items(name='foo', age=30)
    items(name='bar', age=10)
    items(name='bar', age=20)
    items(name='bar', age=30)
    """
    assert len(args) % 2 == 0
    # Extract the filed names as a tuple.
    field_names = args[0::2]
    tpl = collections.namedtuple(name, field_names)
    # Create a generator to yield each combination as a normal tuple.
    iter_base = itertools.product(*args[1::2])
    # Create a list of namedtuples.
    return map(lambda x: tpl(*x), iter_base)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()

# vimquickrun: . ./scripts/activate.sh ; python %
