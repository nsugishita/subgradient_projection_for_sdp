# -*- coding: utf-8 -*-

"""Solve SDPA using MOSEK"""

import argparse
import logging
import os
import pickle

from cpsdppy import config as config_module
from cpsdppy import logging_helper, sdpa
from cpsdppy.sdp_solvers import mosek_solver

logger = logging.getLogger(__name__)

version = "vdev"
tmp_dir = f"tmp/sdpa/{version}/cache"


def run(problem_data, config, disable_cache=False):
    assert config.solver in ["mosek"]
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
        if config.solver == "mosek":
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
    args = parser.parse_args()

    config = config_module.Config()
    config.solver = "mosek"
    config.time_limit = 3600
    config.log_to_logger = 1
    config_module.parse_args(config, args)

    if not config.problem_name:
        raise ValueError("--problem-name is required")

    logging_helper.setup()

    logger.info(f"problem names: {config.problem_name}")
    logger.info(f"step sizes: {config.step_size}")

    print(config._asstr(only_modified=True, shorten=True))

    problem_data = sdpa.read(config)
    run(problem_data, config)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
