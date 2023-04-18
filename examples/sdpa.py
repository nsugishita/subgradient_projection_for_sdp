# -*- coding: utf-8 -*-

"""Solve SDPA using column generation"""

import argparse
import collections
import itertools
import logging
import os
import pickle

import matplotlib.pyplot as plt

from cpsdppy import config as config_module
from cpsdppy import sdpa
from cpsdppy.sdp_solvers import cutting_plane, subgradient_projection

logger = logging.getLogger(__name__)

use_cache = True


def run(type, prefix, problem_data, config):
    assert type in ["subgradient_projection", "cutting_plane"]
    cache_path = f"tmp/sdpa/{type}_{prefix}.pkl"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(cache_path)
    if os.path.exists(cache_path) and use_cache:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if type == "subgradient_projection":
        res = subgradient_projection.run(problem_data, config)
    elif type == "cutting_plane":
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
        default=["theta1"],
    )
    parser.add_argument(
        "--step-sizes",
        type=float,
        nargs="+",
        default=[100],
    )
    config_module.add_arguments(parser)
    args = parser.parse_args()

    base_config = config_module.Config()
    base_config.time_limit = 60
    base_config.parse_args(args)

    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    logging.info(f"problem names: {args.problem_names}")
    logging.info(f"step sizes: {args.step_sizes}")

    setupt = collections.namedtuple("setup", ["cut_type", "lb"])
    iter_base = itertools.product(["lmi", "linear"], [True, False])
    iter = list(map(lambda x: setupt(*x), iter_base))

    def label(setup):
        return setup.cut_type

    def color(setup):
        return "C" + str(
            ["lmi", "linear", "naivelinear"].index(setup.cut_type)
        )

    for problem_name in args.problem_names:
        for step_size in args.step_sizes:
            problem_data = sdpa.read(problem_name)

            results = dict()

            for setup in iter:
                logger.info("- " * 20)
                logger.info(str(setup))
                logger.info("- " * 20)

                config, prefix = update_config(
                    problem_name, base_config, step_size, setup
                )

                results["sub", prefix] = run(
                    "subgradient_projection", prefix, problem_data, config
                )
                if setup.lb:
                    results["cp", prefix] = run(
                        "cutting_plane", prefix, problem_data, config
                    )

            figs = {}
            axes = {}
            figs[True], axes[True] = plt.subplots()
            figs[False], axes[False] = plt.subplots()
            for setup_i, setup in enumerate(iter):
                config, prefix = update_config(
                    problem_name, base_config, step_size, setup
                )
                fig = figs[config.eval_lb_every > 0]
                ax = axes[config.eval_lb_every > 0]

                res = results["sub", prefix]
                y = res["iter_lb_gap"][1:] * 100
                x = res["iter_lb_gap_time"][1:]
                ax.plot(x, y, label=label(setup), color=color(setup))
                y = res["iter_fv_gap"][1:] * 100
                x = res["iter_fv_gap_time"][1:]
                ax.plot(x, y, color=color(setup))
                ax.plot(x, y, color=color(setup))
            for i, (fig, ax) in enumerate(zip(figs.values(), axes.values())):
                ax.legend()
                ax.set_xlabel("elapse (seconds)")
                if ax.get_ylim()[0] < -30:
                    pass
                else:
                    ax.set_ylim(0, 20)
                ax.set_ylabel("suboptimality of bounds (%)")
                path = (
                    f"tmp/sdpa/fig/{problem_name.split('.')[0]}_"
                    f"step_size_{step_size}_walltime_"
                    f"lb_{int(i)}.pdf"
                )
                os.makedirs(os.path.dirname(path), exist_ok=True)
                fig.savefig(path, transparent=True)
                print(path)


def update_config(problem_name, base_config, step_size, setup):
    config = base_config.copy()
    config.step_size = step_size
    if setup.lb:
        config.initial_cut_type = (
            "lmi" if setup.cut_type == "lmi" else "linear"
        )
    else:
        config.initial_cut_type = "none"
    if setup.cut_type == "lmi":
        config.n_linear_cuts_for_unregularised_rmp = 0
        config.n_linear_cuts_for_regularised_rmp = 0
        config.eigenvector_combination_cut = 0
        config.n_lmi_cuts_for_unregularised_rmp = 1
        config.n_lmi_cuts_for_regularised_rmp = 1
    elif setup.cut_type == "naivelinear":
        config.n_linear_cuts_for_unregularised_rmp = 1
        config.n_linear_cuts_for_regularised_rmp = 1
        config.eigenvector_combination_cut = 0
        config.n_lmi_cuts_for_unregularised_rmp = 0
        config.n_lmi_cuts_for_regularised_rmp = 0
    elif setup.cut_type == "linear":
        config.n_linear_cuts_for_unregularised_rmp = 1
        config.n_linear_cuts_for_regularised_rmp = 1
        config.eigenvector_combination_cut = 1
        config.n_lmi_cuts_for_unregularised_rmp = 0
        config.n_lmi_cuts_for_regularised_rmp = 0

    if setup.lb:
        config.eval_lb_every = 1
    else:
        config.eval_lb_every = 0
        config.n_linear_cuts_for_unregularised_rmp = 0
        config.n_lmi_cuts_for_unregularised_rmp = 0
        config.n_linear_cuts_for_unregularised_rmp = 0
        config.n_lmi_cuts_for_unregularised_rmp = 0

    prefix = f"{problem_name.split('.')[0]}_{config.non_default_as_str()}"

    return config, prefix


if __name__ == "__main__":
    main()

# vimquickrun: . ./scripts/activate.sh ; python %
