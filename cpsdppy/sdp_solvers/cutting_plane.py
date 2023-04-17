# -*- coding: utf-8 -*-

"""Cutting-plane solver of SDP"""

import argparse
import collections
import itertools
import logging
import os

import numpy as np
import scipy.sparse

import cpsdppy
from cpsdppy import utils

logger = logging.getLogger(__name__)

use_cache = True


def gap(a: float, b: float, c: float) -> float:
    if np.all(np.isfinite([a, b, c])):
        return (a - b) / np.abs(c)
    else:
        return np.nan


def evaluate_solution(x, problem_data):
    objective_coef = problem_data["objective_coefficient"]
    constr_svec_coefs = problem_data["lmi_svec_constraint_coefficient"]
    constr_svec_offset = problem_data["lmi_svec_constraint_offset"]
    f = objective_coef @ x
    f_gap = gap(
        f,
        problem_data["target_objective"],
        problem_data["target_objective"],
    )
    eigenvalues = []
    eigenvectors = []
    for coef_i in range(len(constr_svec_coefs)):
        matrix = cpsdppy.linalg.svec_inv(
            constr_svec_coefs[coef_i] @ x - constr_svec_offset[coef_i],
            part="f",
        )
        _eigenvalues, _eigenvectors = np.linalg.eigh(matrix)
        eigenvalues.append(_eigenvalues)
        eigenvectors.append(_eigenvectors)
    g = np.array([-_eigenvalues[0] for _eigenvalues in eigenvalues])
    return eval_res_t(f, f_gap, g, eigenvalues, eigenvectors)


eval_res_t = collections.namedtuple(
    "eval_res_t", "f f_gap g eigenvalues eigenvectors"
)


def add_initial_linear_cuts(linear_cuts, constr_coef, constr_offset):
    mat_size = constr_coef[0].shape[0]
    initial_cuts = []
    for i in range(mat_size):
        for j in range(i, mat_size):
            u = np.zeros(mat_size)
            u[i] = 1
            u[j] = 1
            initial_cuts.append(u)

            if i != j:
                u = np.zeros(mat_size)
                u[i] = 1
                u[j] = -1
                initial_cuts.append(u)

    initial_cuts = scipy.sparse.csr_array(np.array(initial_cuts))

    coef, offset = get_linear_cut_coef(
        constr_coef,
        constr_offset,
        initial_cuts,
    )
    linear_cuts.add_linear_cuts(coef=coef, offset=offset)


def add_initial_lmi_cuts(lmi_cuts, constr_coef, constr_offset):
    mat_size = constr_coef[0].shape[0]
    v0 = np.zeros(mat_size)
    v1 = np.zeros(mat_size)

    v0_col = np.repeat(np.arange(mat_size - 1), np.arange(mat_size - 1, 0, -1))
    v0_row = np.arange(v0_col.size)
    v0_val = np.ones_like(v0_row, dtype=float)
    v1_col = np.concatenate(
        [np.arange(i + 1, mat_size) for i in range(mat_size)]
    )
    v1_row = np.arange(v1_col.size)
    v1_val = np.ones_like(v1_row, dtype=float)

    v0_row = np.concatenate([v0_row, v0_row + v0_row.max() + 1])
    v0_col = np.concatenate([v0_col, v0_col])
    v0_val = np.concatenate([v0_val, v0_val])
    v1_row = np.concatenate([v1_row, v1_row + v1_row.max() + 1])
    v1_col = np.concatenate([v1_col, v1_col])
    v1_val = np.concatenate([v1_val, -v1_val])

    v0 = scipy.sparse.csr_matrix(
        (v0_val, (v0_row, v0_col)), shape=(v0_row.size, mat_size), dtype=int
    )
    v1 = scipy.sparse.csr_matrix(
        (v1_val, (v1_row, v1_col)), shape=(v0_row.size, mat_size), dtype=int
    )

    coef, offset = get_lmi_cut_coef(
        constr_coef,
        constr_offset,
        v0,
        v1,
    )
    lmi_cuts.add_lmi_cuts(coef=coef, offset=offset)


def run(problem_data, config):
    n_variables = problem_data["objective_coefficient"].size
    constr_coefs = problem_data["lmi_constraint_coefficient"]
    constr_offsets = problem_data["lmi_constraint_offset"]
    constr_svec_coefs = problem_data["lmi_svec_constraint_coefficient"]
    constr_svec_offset = problem_data["lmi_svec_constraint_offset"]

    journal = utils.IterationJournal()

    journal.register_iteration_items(
        lb=dict(default=np.nan, timing=True),
        best_lb=dict(default=np.nan, timing=True),
        lb_gap=dict(default=np.nan, timing=True),
        ub=dict(default=np.nan, timing=True),
        best_ub=dict(default=np.nan, timing=True),
        ub_gap=dict(default=np.nan, timing=True),
        step_size=dict(default=np.nan, timing=False),
        x=dict(default=np.full(n_variables, np.nan), timing=True),
        fx=dict(default=np.nan, timing=True),
        fx_gap=dict(default=np.nan, timing=True),
        gx=dict(default=np.full(len(constr_svec_coefs), np.nan), timing=True),
        regularised_rmp_n_linear_cuts=dict(default=0, timing=False),
        regularised_rmp_n_lmi_cuts=dict(default=0, timing=False),
        unregularised_rmp_n_linear_cuts=dict(default=-1, timing=False),
        unregularised_rmp_n_lmi_cuts=dict(default=-1, timing=False),
    )

    unregularised_model = cpsdppy.mip_solvers.gurobi.GurobiInterface()
    xlb = problem_data["variable_lb"]
    xub = problem_data["variable_ub"]
    objective_coef = problem_data["objective_coefficient"]
    unregularised_model.add_variables(lb=xlb, ub=xub, obj=objective_coef)

    unreg_linear_cuts = cpsdppy.mip_solver_extensions.LinearCuts(
        unregularised_model
    )
    unreg_lmi_cuts = cpsdppy.mip_solver_extensions.LMICuts(unregularised_model)

    assert config.initial_cut_type in ["linear", "lmi", "none"]
    for coef_i in range(len(constr_svec_coefs)):
        if config.initial_cut_type == "linear":
            add_initial_linear_cuts(
                unreg_linear_cuts, constr_coefs[coef_i], constr_offsets[coef_i]
            )
        elif config.initial_cut_type == "lmi":
            add_initial_lmi_cuts(
                unreg_lmi_cuts, constr_coefs[coef_i], constr_offsets[coef_i]
            )

    best_lb = -np.inf
    best_ub = np.inf

    solver_status = "unknown"

    journal.start_hook()
    timer = utils.timer()

    def remaining_time():
        if config.time_limit is None:
            return np.inf
        if config.time_limit <= 0:
            return np.inf
        if not np.isfinite(config.time_limit):
            return np.inf
        return config.time_limit - timer.walltime

    for iteration in range(1000):
        if 0 < config.iteration_limit <= iteration:
            solver_status = "iteration_limit"
            break
        if 0 < config.time_limit <= timer.walltime:
            solver_status = "time_limit"
            break

        journal.iteration_start_hook(iteration)
        unreg_linear_cuts.iteration = iteration
        unreg_lmi_cuts.iteration = iteration
        lb = -np.inf
        ub = np.inf

        unregularised_model.solve()
        x = unregularised_model.get_solution()[: len(xlb)]
        lb = unregularised_model.get_objective_value()
        best_lb = max(lb, best_lb)

        journal.set_iteration_items(
            unregularised_rmp_n_linear_cuts=unreg_linear_cuts.n,
            unregularised_rmp_n_lmi_cuts=unreg_lmi_cuts.n,
        )

        eval_x = evaluate_solution(x, problem_data)
        most_violated_constr_index = np.argmax(eval_x.g)
        if np.max(eval_x.g) <= config.feas_tol:
            ub = min(ub, eval_x.f)
            best_ub = min(ub, best_ub)

        add_cuts(
            config,
            unreg_linear_cuts,
            unreg_lmi_cuts,
            constr_svec_coefs[most_violated_constr_index],
            constr_svec_offset[most_violated_constr_index],
            x,
            eval_x.eigenvalues[most_violated_constr_index],
            eval_x.eigenvectors[most_violated_constr_index],
            config.n_linear_cuts_for_unregularised_rmp,
            config.n_lmi_cuts_for_unregularised_rmp,
        )

        n_unreg_linear_cuts = unreg_linear_cuts.n
        n_unreg_lmi_cuts = unreg_lmi_cuts.n

        _lb_gap = gap(
            best_lb,
            problem_data["target_objective"],
            problem_data["target_objective"],
        )
        _ub_gap = gap(
            best_ub,
            problem_data["target_objective"],
            problem_data["target_objective"],
        )

        journal.set_iteration_items(
            lb=lb,
            best_lb=best_lb,
            lb_gap=_lb_gap,
            ub=ub,
            best_ub=best_ub,
            ub_gap=_ub_gap,
            x=x,
            fx=eval_x.f,
            fx_gap=eval_x.f_gap,
            gx=eval_x.g,
        )

        head = [
            f"{'it':>3s}",
            f"  {'elapse':>8s}",
            # f"  {'fx':>11s}",
            f"  {'fx_gap (%)':>11s}",
            f"  {'|gx|_inf':>8s}",
            # f"  {'fv':>11s}",
            f"  {'fv_gap (%)':>11s}",
            f"  {'|gv|_inf':>8s}",
            # f"  {'ub_gap (%)':>11s}",
            # f"  {'lb':>11s}",
            f"  {'lb_gap (%)':>11s}",
            f"  {'rcols':>5s}",
            f"  {'ucols':>5s}",
            f"  {'ss':>7s}",
        ]
        # lb_symbol = " "
        n_rcuts = 0
        n_ucuts = n_unreg_linear_cuts + n_unreg_lmi_cuts
        format_number = utils.format_number
        body = [
            f"{iteration:3d}",
            "  ",
            utils.format_elapse(timer.walltime),
            # f"  {format_number(eval_x.f, width=11)}",
            f"  {format_number(eval_x.f_gap * 100, width=11)}",
            f"  {format_number(np.max(eval_x.g), width=8)}",
            # f"  {format_number(eval_v.f, width=11)}",
            f"  {format_number(np.nan, width=11)}",
            f"  {format_number(np.nan, width=8)}",
            # f"  {format_number(_ub_gap * 100, width=11)}",
            # f"  {format_number(lb, width=11)}",
            # lb_symbol,
            f"  {format_number(-_lb_gap * 100, width=11)}",
            f"  {n_rcuts:5d}",
            f"  {n_ucuts:5d}",
            f"  {np.nan:7.1e}",
        ]
        if iteration % (config.log_every * 20) == 0:
            logger.info("".join(head))
        else:
            logger.debug("".join(head))
        logger.info("".join(body))

        lb_closed = np.isfinite(_lb_gap) and (0 <= -_lb_gap <= config.tol)
        solution_found = (
            np.isfinite(eval_x.f_gap)
            and (0 < config.tol)
            and (eval_x.f_gap <= config.tol)
            and (np.max(eval_x.g) <= config.feas_tol)
        )
        assert config.termination_criteria in [
            "lb_and_solution",
            "solution",
            "lb",
        ]
        if config.termination_criteria == "lb_and_solution":
            if lb_closed and solution_found:
                solver_status = "solved"
                break
        elif config.termination_criteria == "solution":
            if solution_found:
                solver_status = "solved"
                break
        else:
            if lb_closed:
                solver_status = "solved"
                break

    result = dict()
    result.update(
        dict(
            algorithm="subgradient_projection",
            hostname=os.uname()[1],
            solver_status=solver_status,
            lb=np.nanmax(journal.get_all("lb")),
            walltime=timer.walltime,
            proctime=timer.proctime,
            n_iterations=len(journal.get_all("lb")),
            target_objective=problem_data["target_objective"],
            stepsize=config.step_size,
            time_limit=config.time_limit,
            iteration_limit=config.iteration_limit,
        )
    )
    journal.dump_data(out=result)
    timer.dump_data(out=result)
    for key in result:
        result[key] = np.asarray(result[key])
    return result


def add_cuts(
    config,
    linear_cuts,
    lmi_cuts,
    constr_svec_coef,
    constr_svec_offset,
    x,
    w,
    v,
    n_linear_cuts,
    n_lmi_cuts,
):
    # TODO Improve efficiency using initialisation routine.

    for i in range(n_lmi_cuts):
        v0 = v[:, 2 * i]
        v1 = v[:, 2 * i + 1]
        v0v0t = cpsdppy.linalg.svec(v0[:, None] @ v0[None, :])
        v0v1t = (
            cpsdppy.linalg.svec(
                v0[:, None] @ v1[None, :] + v1[:, None] @ v0[None, :]
            )
            / 2
        )
        v1v1t = cpsdppy.linalg.svec(v1[:, None] @ v1[None, :])

        cut_coef = np.stack(
            [
                v0v0t @ constr_svec_coef,
                v0v1t @ constr_svec_coef,
                v1v1t @ constr_svec_coef,
            ]
        )
        cut_offset = np.array(
            [
                v0v0t @ constr_svec_offset,
                v0v1t @ constr_svec_offset,
                v1v1t @ constr_svec_offset,
            ]
        )
        lmi_cuts.add_lmi_cuts(coef=cut_coef, offset=cut_offset)

    for i in range(n_linear_cuts):
        v0 = v[:, i + 2 * n_lmi_cuts]
        v0v0t = cpsdppy.linalg.svec(v0[:, None] @ v0[None, :])
        cut_coef = v0v0t @ constr_svec_coef
        cut_offset = v0v0t @ constr_svec_offset
        linear_cuts.add_linear_cuts(coef=-cut_coef, offset=-cut_offset)

    if config.eigenvector_combination_cut:
        v0 = np.sum(np.clip(w, None, 0.0) * v, axis=1)
        norm = np.linalg.norm(v0)
        if norm != 0:
            v0v0t = cpsdppy.linalg.svec(v0[:, None] @ v0[None, :])
            cut_coef = v0v0t @ constr_svec_coef
            cut_offset = v0v0t @ constr_svec_offset
            linear_cuts.add_linear_cuts(coef=-cut_coef, offset=-cut_offset)


def get_linear_cut_coef(constr_coef, constr_offset, v):
    assert isinstance(v, scipy.sparse.spmatrix)

    if (
        isinstance(constr_coef, list)
        and (len(constr_coef) > 0)
        and isinstance(constr_coef[0], scipy.sparse.spmatrix)
    ):
        constr_coef = np.array([x.toarray() for x in constr_coef])
        constr_offset = constr_offset.toarray()

    if v.shape[1] == 1:
        v = v.T

    n_cuts = v.shape[0]

    v = v.tocsr()

    coefs = []
    offsets = []

    for i in range(n_cuts):
        col = v.indices[v.indptr[i] : v.indptr[i + 1]]
        val = v.data[v.indptr[i] : v.indptr[i + 1]]
        n = col.size

        col0 = np.repeat(col, n)
        val0 = np.repeat(val, n)
        col1 = np.tile(col, n)
        val1 = np.tile(val, n)

        coefs.append(
            np.sum(
                val0[None, :] * val1[None, :] * constr_coef[:, col0, col1],
                axis=1,
            )
        )
        offsets.append(
            np.sum(
                val0[None, :] * val1[None, :] * constr_offset[col0, col1],
            )
        )

    coefs = np.stack(coefs)
    offsets = np.array(offsets)
    np.testing.assert_equal(coefs.ndim, 2)
    np.testing.assert_equal(offsets.shape, (n_cuts,))

    return -coefs, -offsets


def get_lmi_cut_coef(constr_coef, constr_offset, v0, v1):
    assert isinstance(v0, scipy.sparse.spmatrix)

    if (
        isinstance(constr_coef, list)
        and (len(constr_coef) > 0)
        and isinstance(constr_coef[0], scipy.sparse.spmatrix)
    ):
        constr_coef = np.array([x.toarray() for x in constr_coef])
        constr_offset = constr_offset.toarray()

    if v0.shape[1] == 1:
        v0 = v0.T
        v1 = v1.T

    n_cuts = v0.shape[0]

    v0 = v0.tocsr()
    v1 = v1.tocsr()

    coefs = []
    offsets = []

    for i in range(n_cuts):
        col0 = v0.indices[v0.indptr[i] : v0.indptr[i + 1]]
        val0 = v0.data[v0.indptr[i] : v0.indptr[i + 1]]
        n0 = col0.size
        col1 = v1.indices[v1.indptr[i] : v1.indptr[i + 1]]
        val1 = v1.data[v1.indptr[i] : v1.indptr[i + 1]]
        n1 = col1.size

        # TODO CHECK
        col0 = np.repeat(col0, n1)
        val0 = np.repeat(val0, n1)
        col1 = np.tile(col1, n0)
        val1 = np.tile(val1, n0)

        # TODO CHECK
        coefs.extend(
            [
                np.sum(
                    val0[:, None] * val0[:, None] * constr_coef[:, col0, col0],
                    axis=1,
                ),
                np.sum(
                    val0[:, None] * val1[:, None] * constr_coef[:, col0, col1],
                    axis=1,
                ),
                np.sum(
                    val1[:, None] * val1[:, None] * constr_coef[:, col1, col1],
                    axis=1,
                ),
            ]
        )
        offsets.extend(
            [
                np.sum(
                    val0[:, None] * val0[:, None] * constr_offset[col0, col0],
                ),
                np.sum(
                    val0[:, None] * val1[:, None] * constr_offset[col0, col1],
                ),
                np.sum(
                    val1[:, None] * val1[:, None] * constr_offset[col1, col1],
                ),
            ]
        )

    coefs = np.array(coefs).reshape(n_cuts, 3, -1)
    offsets = np.array(offsets).reshape(n_cuts, 3)

    return coefs, offsets


def main() -> None:
    """Run the main routine of this script"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem-names",
        type=str,
        nargs="+",
        default=["theta1", "theta2", "theta3"],
    )
    parser.add_argument(
        "--step-sizes",
        type=float,
        nargs="+",
        default=[100, 1000],
    )
    cpsdppy.config.add_arguments(parser)
    args = parser.parse_args()

    base_config = cpsdppy.config.Config()
    base_config.iteration_limit = 4
    base_config.time_limit = 60
    base_config.parse_args(args)

    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    logging.info(f"problem names: {args.problem_names}")
    logging.info(f"step sizes: {args.step_sizes}")

    setupt = collections.namedtuple("setup", ["cut_type", "lb"])
    iter_base = itertools.product(["lmi", "linear"], [True])
    iter = list(map(lambda x: setupt(*x), iter_base))

    def label(setup):
        return setup.cut_type

    def color(setup):
        return "C" + str(
            ["lmi", "linear", "naivelinear"].index(setup.cut_type)
        )

    for problem_name in args.problem_names:
        for step_size in args.step_sizes:
            problem_data = cpsdppy.sdpa.read(problem_name)

            results = dict()

            for setup in iter:
                logger.info("- " * 20)
                logger.info(str(setup))
                logger.info("- " * 20)

                config, prefix = update_config(
                    problem_name, base_config, step_size, setup
                )

                results[prefix] = run(problem_data, config)


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
