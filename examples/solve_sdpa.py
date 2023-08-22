# -*- coding: utf-8 -*-

"""Solve SDPA

This scripts run the subgradient projection solver or MOSEK
to solve a test instance in SDPLIB.

The following two functions are defined in this script.

- run(config, dir):
    Run a solver in the same process.
- run_subprocess(config, dir):
    Run a solver on a subprocess. Returns `returncode` and
    the object returned by the solver.

Additionally, one can run a solver from the command line.

```
$ python examples/solve_sdpa.py --problem-name theta1 --solver mosek
```

"""

import argparse
import logging
import os
import pickle
import subprocess
import typing

from cpsdppy import config as config_module
from cpsdppy import logging_helper, sdpa
from cpsdppy.sdp_solvers import cosmo, mosek, subgradient_projection

logger = logging.getLogger(__name__)


def run(config, dir):
    """Run the solver on the current process"""
    paths = _get_paths(config, dir)
    if config.dry_run:
        cache_path = log_path = None
    else:
        cache_path = paths["cache"]
        log_path = paths["log"]
    logger.info("result are saved in:")
    logger.info(cache_path)
    logger.info("log messages are saved in:")
    logger.info(log_path)

    if (not config.dry_run) and os.path.exists(cache_path):
        # Cache files exist. Retrieve the cached result.
        logger.info("cache found")
        with open(cache_path, "rb") as f:
            result = pickle.load(f)

    else:
        # Cache not found. We need to run the solver.
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with logging_helper.save_log(log_path):
            result = _run_impl(config)

        if not config.dry_run:
            # Create a cache file and save the result.
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)

    return result


def run_subprocess(config, dir):
    """Run the solver on a subprocess"""
    if config.dry_run:
        returncode_path = None
    else:
        returncode_path = _get_paths(config, dir)["returncode"]

    if (not config.dry_run) and os.path.exists(returncode_path):
        # Cache files exist. Retrieve the cached returncode.
        with open(returncode_path, "r") as f:
            returncode = int(f.read())

    else:
        # Cache not found. Run a subprocess and wait for it.
        returncode = _run_subprocess_impl(config, dir)

        if not config.dry_run:
            # Create a cache file and save the returncode.
            os.makedirs(os.path.dirname(returncode_path), exist_ok=True)
            with open(returncode_path, "w") as f:
                f.write(str(returncode))

    # Load a cached result if exists.
    result = _load_result(config, dir)
    return returncode, result


missing: dict = {}


def load_result(config, dir, default=missing):
    """Run the solver on a subprocess"""
    returncode_path = _get_paths(config, dir)["returncode"]
    if os.path.exists(returncode_path):
        # Cache files exist. Retrieve the cached returncode.
        with open(returncode_path, "r") as f:
            returncode = int(f.read())
    else:
        returncode = None

    # Load a cached result if exists.
    result = _load_result(config, dir, default=default)

    if result is missing:
        raise FileNotFoundError(
            config._asstr(only_modified=True, shorten=True)
        )

    return returncode, result


def _run_impl(config):
    problem_data = sdpa.read(config)

    if config.solver == "subgradient_projection":
        res = subgradient_projection.run(problem_data, config)
    elif config.solver == "mosek":
        res = mosek.run(problem_data, config)
    elif config.solver == "cosmo":
        res = cosmo.run(problem_data, config)
    elif config.solver == "":
        raise ValueError("config.solver is missing")
    else:
        raise ValueError(f"unknown solver '{config.solver}'")

    return res


def _run_subprocess_impl(config, dir):
    command = f"python examples/solve_sdpa.py --dir {dir} "
    for key, value in config._asdict(only_modified=True).items():
        value = str(value)
        if "~" in value:
            value = f'"{value}"'
        command += f"--{key.replace('_', '-')} {str(value)} "

    ret = subprocess.run(command, shell=True)
    return ret.returncode


def _load_result(config, dir, default=missing):
    """Load a cached result"""
    cache_path = _get_paths(config, dir)["cache"]

    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError as e:
        if default is missing:
            raise e from None
        else:
            return default


def _get_paths(config, dir) -> typing.Dict[str, str]:
    config_str = config._asstr(only_modified=True, shorten=True)
    return dict(
        cache=f"{dir}/{config_str}.pkl",
        log=f"{dir}/{config_str}.txt",
        returncode=f"{dir}/{config_str}_returncode.txt",
    )


def main() -> None:
    """Run the entry point of this script"""
    parser = argparse.ArgumentParser()
    config_module.add_arguments(parser)
    parser.add_argument(
        "--dir",
        type=str,
    )
    args = parser.parse_args()

    config = config_module.Config()
    config._parse_args()

    if args.dir is None:
        config.dry_run = 1

    logging_helper.setup()

    if config.problem_name == "":
        raise ValueError("--problem-name is required")
    if config.solver == "":
        raise ValueError("--solver is required")

    run(config, dir=args.dir)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
