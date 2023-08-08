# -*- coding: utf-8 -*-

"""Run COSMO"""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def run(problem_data, config):
    """Run COSMO

    Returns
    -------
    returncode : int
    walltime : float
    n_iterations : int
    """
    prefix = os.environ.get("PREFIX", "/usr")
    command = (
        f"{prefix}/bin/julia --project=juliaenv "
        "-e 'include(\"examples/run_cosmo.jl\");' -- theta1 1e-3"
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    walltime = -1.0
    n_iterations = -1

    proc = subprocess.Popen(
        args=command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf8",
        env=env,
        shell=1,
    )
    while True:
        for line in iter(proc.stdout.readline, ""):
            for _line in line[:-1].split("\n"):
                logger.info(_line)
            if line.startswith("walltime:"):
                walltime = float(line.split()[1])
            if line.startswith("n_iterations:"):
                n_iterations = int(line.split()[1])
        proc.poll()  # Set return code
        returncode = proc.returncode
        if returncode is not None:
            if returncode == 0:
                return dict(walltime=walltime, n_iterations=n_iterations)
            else:
                raise SystemExit(returncode)
