# -*- coding: utf-8 -*-

"""Utils for mosek"""

import logging
import time

import numpy as np


def solve(model, config={}):
    """Run mosek

    `config` may has the following items.

    - n_threads : int
        Number of threads.
    - tol : float, default 0
        This is used to set "intpntCoTolPfeas". This is ignored when
        `mosek.tol` is given.
    - feas_tol : float, default 0
        This is used to set "intpntCoTolDfeas", "intpntCoTolMuRed"
        and "intpntCoTolRelGap". This is ignored when `mosek.tol`
        is given.
    - mosek.tol : float, default 0
        This is used to set "intpntCoTolDfeas", "intpntCoTolMuRed",
        "intpntCoTolPfeas" and "intpntCoTolRelGap".
    - time_limit : float, default 0
        If positive, this is used to set "optimizerMaxTime".
    - iteration_limit : float, default 0
        If positive, this is used to set "intpntMaxIterations".

    The returned dict will contain the following items.

    - algorithm : str
        This is always set to be "mosek".
    - n_iterations : int
    - walltime : float
    - primal_objective : float
    - dual_objective : float
    - iter_pfeas : list of float
    - iter_dfeas : list of float
    - iter_greas : list of float
    - iter_prstatus : list of float
    - iter_primal_objective : list of float
    - iter_dual_objective : list of float
    - iter_mu : list of float
    - iter_mosektime : list of float
    - iter_walltime : list of float

    Parameters
    ----------
    model : mosek model
    config : dict

    Returns
    -------
    res : dict
    """
    import mosek.fusion

    logger = logging.getLogger(__name__)

    model.setSolverParam("numThreads", int(config.n_threads))

    optimality_tol = config.tol
    if optimality_tol > 0:
        model.setSolverParam("intpntCoTolRelGap", optimality_tol)
    feasibility_tol = config.feas_tol
    if feasibility_tol > 0:
        model.setSolverParam("intpntCoTolDfeas", feasibility_tol)
        model.setSolverParam("intpntCoTolMuRed", feasibility_tol)
        model.setSolverParam("intpntCoTolPfeas", feasibility_tol)

    time_limit = config.time_limit
    if time_limit > 0:
        model.setSolverParam("optimizerMaxTime", time_limit)
    iteration_limit = config.iteration_limit
    if iteration_limit > 0:
        model.setSolverParam("intpntMaxIterations", iteration_limit)

    log_to_logger = config.log_to_logger
    log_to_stdout = config.log_to_stdout

    class capture_log:
        def __init__(self):
            self.log = []
            self.log_walltime = []

        def write(self, line):
            self.log.append(line)
            self.log_walltime.append(time.perf_counter())
            if log_to_logger:
                logger.info(line[:-1])
            else:
                logger.debug(line[:-1])
            if log_to_stdout:
                print(line[:-1])

        def flush(self):
            pass

    capture = capture_log()
    model.setLogHandler(capture)

    starttime = time.perf_counter()

    model.solve()
    walltime = time.perf_counter() - starttime

    parsed = _parse_mosek_log(capture.log, capture.log_walltime)
    parsed["primal_objective"] = parsed.pop("pobj")
    parsed["dual_objective"] = parsed.pop("dobj")

    res = {}
    res["algorithm"] = "mosek"
    res["n_iterations"] = len(parsed["ite"])
    res["walltime"] = walltime
    res["primal_objective"] = res["dual_objective"] = float("nan")

    res["problem_status"] = str(model.getProblemStatus())
    res["primal_solution_status"] = str(model.getPrimalSolutionStatus())
    res["dual_solution_status"] = str(model.getDualSolutionStatus())

    try:
        res["primal_objective"] = model.primalObjValue()
    except mosek.fusion.SolutionError:
        pass
    try:
        res["dual_objective"] = model.dualObjValue()
    except mosek.fusion.SolutionError:
        pass

    try:
        primal_solution = []
        dual_solution = []
        for i in range(1, 1000):
            var = model.getVariable(i)
            if var is None:
                break
            primal_solution.append(np.asarray(var.level()).ravel())
            dual_solution.append(np.asarray(var.dual()).ravel())
        if primal_solution:
            primal_solution = np.concatenate(primal_solution)
            dual_solution = np.concatenate(dual_solution)
        else:
            primal_solution = np.array([])
            dual_solution = np.array([])
        res["primal_solution"] = primal_solution
        res["dual_solution"] = dual_solution

    except mosek.fusion.SolutionError:
        pass

    for key, value in parsed.items():
        res["iter_" + key] = value

    res.pop("iter_ite")

    return res


def _parse_mosek_log(log, log_walltime):
    """Parse mosek log and extract data such as primal and dual objective

    This parses log of mosek's barrier method.
    The returned dict will contain the following items.

    - ite
    - pfeas
    - dfeas
    - greas
    - prstatus
    - pobj
    - dobj
    - mu
    - mosektime
    - walltime

    Parameters
    ----------
    log : list of str
        Each item is a line of mosek log
    log_walltime : list of float
        Each item is a walltime when the corresponding mosek log was emitted.

    Returns
    -------
    res : dict
    """
    t0 = log_walltime[0]
    log_walltime = [t - t0 for t in log_walltime]

    main_iteration = False
    lines = []
    lines_walltime = []
    for line, t in zip(log, log_walltime):
        if not main_iteration:
            if line.startswith("ITE"):
                main_iteration = True
            continue
        if line.startswith("Optimizer terminated"):
            break
        lines.append(list(map(float, line.split())))
        lines_walltime.append(t)
    keys = [
        "ite",
        "pfeas",
        "dfeas",
        "greas",
        "prstatus",
        "pobj",
        "dobj",
        "mu",
        "time",
    ]
    res = {}
    for i, key in enumerate(keys):
        res[key] = [line[i] for line in lines]
    res["mosektime"] = res.pop("time")
    res["walltime"] = lines_walltime
    for key in res:
        res[key] = np.asarray(res[key])
    return res
