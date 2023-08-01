# -*- coding: utf-8 -*-

"""Solve SDPA using column generation"""

import argparse
import collections
import itertools
import logging
import os
import pickle

import numpy as np
import pandas as pd

from cpsdppy import config as config_module
from cpsdppy import logging_helper, sdpa
from cpsdppy.sdp_solvers import subgradient_projection

logger = logging.getLogger(__name__)

version = "v6"
tmp_dir = f"tmp/sdpa/{version}/cache"


def run(problem_data, config, disable_cache):
    assert config.solver in ["subgradient_projection"]
    cache_path = (
        f"{tmp_dir}/data/{config._asstr(only_modified=True, shorten=True)}.pkl"
    )
    log_path = (
        f"{tmp_dir}/data/{config._asstr(only_modified=True, shorten=True)}.txt"
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    logger.info("result are saved in:")
    logger.info(cache_path)
    logger.info("log messages are saved in:")
    logger.info(log_path)
    if os.path.exists(cache_path) and (not disable_cache):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    with logging_helper.save_log(log_path):
        if config.solver == "subgradient_projection":
            res = subgradient_projection.run(problem_data, config)
        else:
            raise ValueError(f"unknown solver '{config.solver}'")

    if not disable_cache:
        with open(cache_path, "wb") as f:
            pickle.dump(res, f)
    return res


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
        "--step-sizes",
        type=float,
        nargs="+",
        default=[1],
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )
    config_module.add_arguments(parser)
    args = parser.parse_args()

    if args.smoke_test:
        args.disable_cache = True

    base_config = config_module.Config()
    base_config.solver = "subgradient_projection"
    config_module.parse_args(base_config, args)

    logging_helper.setup()

    logger.info(f"problem names: {args.problem_names}")
    logger.info(f"step sizes: {args.step_sizes}")

    def setup_filter(setup):
        if setup.n_linear_cuts == 0:
            if setup.eigen_comb_cut == 0:
                return False
        return True

    if not args.smoke_test:
        setups = list(
            namedtuples_from_product(
                "setup",
                "problem_name",
                args.problem_names,
                "tol",
                [1e-2, 1e-3],
                "step_size",
                args.step_sizes,
                "n_linear_cuts",
                [0, 1],
                "eigen_comb_cut",
                [0, 1],
            )
        )
    else:
        setups = list(
            namedtuples_from_product(
                "setup",
                "problem_name",
                args.problem_names,
                "tol",
                [1e-3],
                "step_size",
                args.step_sizes,
                "n_linear_cuts",
                [1],
                "eigen_comb_cut",
                [1],
            )
        )
    setups = list(filter(setup_filter, setups))

    results: list = []

    for setup in setups:
        logger.info("- " * 20)
        logger.info(str(setup))
        logger.info("- " * 20)

        config = update_config(base_config, setup)
        problem_data = sdpa.read(config)

        results.append(
            (
                config._astuple(shorten=True),
                run(problem_data, config, args.disable_cache),
            )
        )

        summary(results)


def summary(results):
    """Print summary of results

    This takes a list of 2-tuple, `(configurations, results)`.
    `configurations` is a tuple of pairs `(configuration, value)`,
    while `resuls` is a dictionary with `walltime` and `n_iterations`.
    """
    if len(results) <= 1:
        return
    data = tuple(
        k
        + (
            ("walltime", v["walltime"]),
            ("n_iterations", v["n_iterations"]),
        )
        for k, v in results
    )
    df = pd.DataFrame.from_records([{k: v for k, v in x} for x in data])
    dropped = []
    for column in list(df.columns):
        if np.unique(df[column]).size == 1:
            dropped.append(column)
    df.drop(columns=dropped, inplace=True)
    for tpls, _ in results:
        index = [k for k, _ in tpls if k not in dropped]
        break
    df = df.set_index(index)
    df = df.sort_index()
    df["walltime"] = df["walltime"].astype(float)
    df["walltime"] = np.round(df["walltime"].values, 2)
    with pd.option_context("display.max_rows", 999):
        print(df)
        with open(f"{tmp_dir}/summary.txt", "w") as f:
            f.write(df.to_string())
            f.write("\n")
    df.to_csv(f"{tmp_dir}/summary.csv")


def update_config(base_config, setup):
    config = config_module.copy(base_config)
    for key, value in setup._asdict().items():
        setattr(config, key, value)
    return config


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
