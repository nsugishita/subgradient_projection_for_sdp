# -*- coding: utf-8 -*-

"""Solve SDPA using MOSEK"""

import argparse
import logging
import os
import pickle

from cpsdppy import config as config_module
from cpsdppy import logging_helper, sdpa
from cpsdppy.sdp_solvers import mosek_solver, subgradient_projection

logger = logging.getLogger(__name__)


def run(config, dir, disable_cache=False):
    cache_path = f"{dir}/{config._asstr(only_modified=True, shorten=True)}.pkl"
    log_path = f"{dir}/{config._asstr(only_modified=True, shorten=True)}.txt"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    logger.info("result are saved in:")
    logger.info(cache_path)
    logger.info("log messages are saved in:")
    logger.info(log_path)
    if os.path.exists(cache_path) and (not disable_cache):
        logger.info("cache found")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    problem_data = sdpa.read(config)

    with logging_helper.save_log(log_path):
        if config.solver == "subgradient_projection":
            res = subgradient_projection.run(problem_data, config)
        elif config.solver == "mosek":
            res = mosek_solver.run(problem_data, config)
        else:
            raise ValueError(f"unknown solver '{config.solver}'")

    if not disable_cache:
        with open(cache_path, "wb") as f:
            pickle.dump(res, f)
    return res


def main() -> None:
    """Run the main routine of this script"""
    parser = argparse.ArgumentParser()
    config_module.add_arguments(parser)
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
    )
    args = parser.parse_args()

    config = config_module.Config()
    config._parse_args()

    if not config.problem_name:
        raise ValueError("--problem-name is required")
    if not config.solver:
        raise ValueError("--solver is required")

    logging_helper.setup()

    logger.info(f"problem names: {config.problem_name}")
    logger.info(f"step sizes: {config.step_size}")

    run(config, dir=args.dir, disable_cache=args.disable_cache)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
