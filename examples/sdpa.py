# -*- coding: utf-8 -*-

"""Solve SDPA using column generation"""

import argparse
import collections
import itertools
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cpsdppy import config as config_module
from cpsdppy import logging_helper, sdpa
from cpsdppy.sdp_solvers import cutting_plane, subgradient_projection

logger = logging.getLogger(__name__)

use_cache = True

# TODO Print time, hostcomputer etc at the beginning.
# TODO Simplify Config.


def run(problem_data, config):
    assert config.solver in ["subgradient_projection", "cutting_plane"]
    cache_path = (
        f"tmp/sdpa/cache/{config._asstr(only_modified=True, shorten=True)}.pkl"
    )
    log_path = (
        f"tmp/sdpa/cache/{config._asstr(only_modified=True, shorten=True)}.txt"
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    logger.info("result are saved in:")
    logger.info(cache_path)
    logger.info("log messages are saved in:")
    logger.info(log_path)
    if os.path.exists(cache_path) and use_cache:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    with logging_helper.save_log(log_path):
        if config.solver == "subgradient_projection":
            res = subgradient_projection.run(problem_data, config)
        elif config.solver == "cutting_plane":
            res = cutting_plane.run(problem_data, config)
        else:
            raise ValueError

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
            # "arch0",
            # "arch2",
            # "arch4",
            # "arch8",
            # "control1",
            # "control2",
            # "control3",
            # "control4",
            # "control5",
            # "control6",
            # "control7",
            # "control8",
            # "control9",
            # "control10",
            # "control11",
            # "eqaulG11",
            # "equalG51",
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
            # "hinf1",
            # "hinf2",
            # "hinf3",
            # "hinf4",
            # "hinf5",
            # "hinf6",
            # "hinf7",
            # "hinf8",
            # "hinf9",
            # "hinf10",
            # "hinf11",
            # "hinf12",
            # "hinf13",
            # "hinf14",
            # "hinf15",
            # "maxG11",
            # "maxG32",
            # "maxG51",
            # "maxG55",
            # "maxG60",
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
            # "qap5",
            # "qap6",
            # "qap7",
            # "qap8",
            # "qap9",
            # "qap10",
            # "qpG11",
            # "qpG51",
            # "ss30",
            "theta1",
            "theta2",
            "theta3",
            # "theta4",
            # "theta5",
            # "theta6",
            # "thetaG11",
            # "thetaG51",
            # "truss1",
            # "truss2",
            # "truss3",
            # "truss4",
            # "truss5",
            # "truss6",
            # "truss7",
            # "truss8",
        ],
    )
    parser.add_argument(
        "--step-sizes",
        type=float,
        nargs="+",
        # default=[100, 1000],
        default=[1],
    )
    config_module.add_arguments(parser)
    args = parser.parse_args()

    base_config = config_module.Config()
    base_config.time_limit = 600
    # base_config.iteration_limit = 30
    base_config.memory = 5
    config_module.parse_args(base_config, args)

    logging_helper.setup()

    logger.info(f"problem names: {args.problem_names}")
    logger.info(f"step sizes: {args.step_sizes}")

    def setup_filter(setup):
        if setup.n_cuts == 0:
            if setup.eigen_comb_cut == 0:
                return False
            if setup.cut_type == "lmi":
                return False
        if setup.lmi_cuts_from_unique_vectors == 0:
            if setup.cut_type == "linear":
                return False
            if setup.n_cuts <= 1:
                return False
        return True

    setups = list(
        namedtuples_from_product(
            "setup",
            "problem_name",
            args.problem_names,
            "step_size",
            args.step_sizes,
            "cut_type",
            ["linear", "lmi"],
            "n_cuts",
            [0, 1, 2, 4],
            "eigen_comb_cut",
            [0, 1],
            "lmi_cuts_from_unique_vectors",
            [1],
            "lb",
            [False],
        )
    )
    setups = list(filter(setup_filter, setups))

    def label(setup):
        return setup.cut_type

    def color(setup):
        return "C" + str(
            ["lmi-solo", "lmi", "linear", "linear-solo"].index(setup.cut_type)
        )

    results: list = []

    for setup in setups:
        logger.info("- " * 20)
        logger.info(str(setup))
        logger.info("- " * 20)

        config = update_config(base_config, setup)
        problem_data = sdpa.read(config)

        config.solver = "subgradient_projection"
        results.append(
            (config._astuple(shorten=True), run(problem_data, config))
        )

        if setup.lb:
            config.solver = "cutting_plane"
            results.append(
                (config._astuple(shorten=True), run(problem_data, config))
            )

        summary(results)

    raise SystemExit

    figs = {}
    axes = {}

    fig_keys = list(
        namedtuples_from_product(
            "fig_key",
            "problem_name",
            args.problem_names,
            "step_size",
            args.step_sizes,
            "lb",
            [True, False],
        )
    )
    for key in fig_keys:
        figs[key], axes[key] = plt.subplots()

    def from_setup_to_fig_key(setup):
        return setup.problem_name, setup.step_size, setup.lb

    for setup in setups:
        fig_key = from_setup_to_fig_key(setup)
        fig, ax = figs[fig_key], axes[fig_key]

        config = update_config(base_config, setup)
        config.solver = "subgradient_projection"
        res = results[config._astuple()]

        y = res["iter_lb_gap"][1:] * 100
        x = res["iter_lb_gap_time"][1:]
        ax.plot(x, y, label=label(setup), color=color(setup))
        y = res["iter_fv_gap"][1:] * 100
        x = res["iter_fv_gap_time"][1:]
        ax.plot(x, y, color=color(setup))
        ax.plot(x, y, color=color(setup))
    for fig_key in fig_keys:
        fig, ax = figs[fig_key], axes[fig_key]
        ax.legend()
        ax.set_xlabel("elapse (seconds)")
        if fig_key.lb:
            pass
        else:
            ax.set_ylim(0, 20)
        ax.set_ylabel("suboptimality of bounds (%)")
        path = (
            f"tmp/sdpa/fig/{fig_key.problem_name.split('.')[0]}_"
            f"step_size_{fig_key.step_size}_walltime_"
            f"lb_{int(fig_key.lb)}.pdf"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, transparent=True)
        print(path)


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
    df.to_csv("tmp/summary.csv")


def update_config(base_config, setup):
    config = config_module.copy(base_config)
    config.problem_name = setup.problem_name
    config.step_size = setup.step_size
    if setup.lb:
        config.initial_cut_type = (
            "lmi" if "lmi" in setup.cut_type else "linear"
        )
    else:
        config.initial_cut_type = "none"
    n = setup.n_cuts
    config.lmi_cuts_from_unique_vectors = setup.lmi_cuts_from_unique_vectors
    config.eigen_comb_cut = setup.eigen_comb_cut
    if setup.cut_type == "lmi":
        config.n_linear_cuts = 0
        config.n_lmi_cuts = n
    elif setup.cut_type == "linear":
        config.n_linear_cuts = n
        config.n_lmi_cuts = 0

    if setup.lb:
        config.eval_lb_every = 1
    else:
        config.eval_lb_every = 0

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
