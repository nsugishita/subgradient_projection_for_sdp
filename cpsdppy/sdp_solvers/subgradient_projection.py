# -*- coding: utf-8 -*-

"""Subgradient projection solver"""

# TODO Improve performance of the solver. Check step size adjustament.
# TODO Record time of subroutines.

import argparse
import collections
import itertools
import logging
import os
import pickle

import matplotlib.pyplot as plt
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


def run_subgradient_projection(prefix, problem_data, config):
    cache_path = f"tmp/sdpa/{prefix}.pkl"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(cache_path)
    if os.path.exists(cache_path) and use_cache:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    res = _run_subgradient_projection_impl(problem_data, config)

    with open(cache_path, "wb") as f:
        pickle.dump(res, f)
    return res


def _run_subgradient_projection_impl(problem_data, config):
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
        v=dict(default=np.full(n_variables, np.nan), timing=True),
        fv=dict(default=np.nan, timing=True),
        fv_gap=dict(default=np.nan, timing=True),
        gv=dict(default=np.full(len(constr_svec_coefs), np.nan), timing=True),
        regularised_rmp_n_linear_cuts=dict(default=-1, timing=False),
        regularised_rmp_n_lmi_cuts=dict(default=-1, timing=False),
        unregularised_rmp_n_linear_cuts=dict(default=-1, timing=False),
        unregularised_rmp_n_lmi_cuts=dict(default=-1, timing=False),
    )

    regularised_model = cpsdppy.mip_solvers.gurobi.GurobiInterface()
    unregularised_model = cpsdppy.mip_solvers.gurobi.GurobiInterface()
    xlb = problem_data["variable_lb"]
    xub = problem_data["variable_ub"]
    objective_coef = problem_data["objective_coefficient"]
    regularised_model.add_variables(lb=xlb, ub=xub, obj=objective_coef)
    unregularised_model.add_variables(lb=xlb, ub=xub, obj=objective_coef)

    reg = cpsdppy.mip_solver_extensions.MoreuYoshidaRegularisation(
        regularised_model, config=config
    )
    reg.step_size = config.step_size
    reg_linear_cuts = cpsdppy.mip_solver_extensions.LinearCuts(
        regularised_model
    )
    unreg_linear_cuts = cpsdppy.mip_solver_extensions.LinearCuts(
        unregularised_model
    )
    reg_lmi_cuts = cpsdppy.mip_solver_extensions.LMICuts(regularised_model)
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

    x = np.zeros(n_variables)

    solver_status = "unknown"

    journal.start_hook()
    timer = utils.timer()
    step_size_manager = StepSizeManager(config)

    def remaining_time():
        if config.time_limit is None:
            return np.inf
        if config.time_limit <= 0:
            return np.inf
        if not np.isfinite(config.time_limit):
            return np.inf
        return config.time_limit - timer.walltime

    eval_res_t = collections.namedtuple(
        "eval_res_t", "f f_gap g eigenvalues eigenvectors"
    )

    def evaluate_solution(x):
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

    for iteration in range(1000):
        if 0 < config.iteration_limit <= iteration:
            solver_status = "iteration_limit"
            break
        if 0 < config.time_limit <= timer.walltime:
            solver_status = "time_limit"
            break

        journal.iteration_start_hook(iteration)
        reg_linear_cuts.iteration = iteration
        reg_lmi_cuts.iteration = iteration
        unreg_linear_cuts.iteration = iteration
        unreg_lmi_cuts.iteration = iteration
        lb = -np.inf
        ub = np.inf

        if (config.eval_lb_every > 0) and (
            iteration % config.eval_lb_every == 0
        ):
            unregularised_model.solve()
            lb = unregularised_model.get_objective_value()
            best_lb = max(lb, best_lb)

        journal.set_iteration_items(
            unregularised_rmp_n_linear_cuts=unreg_linear_cuts.n,
            unregularised_rmp_n_lmi_cuts=unreg_lmi_cuts.n,
        )

        eval_x = evaluate_solution(x)
        most_violated_constr_index = np.argmax(eval_x.g)
        if np.max(eval_x.g) <= config.feas_tol:
            ub = min(ub, eval_x.f)
            best_ub = min(ub, best_ub)

        if config.eval_lb_every > 0:
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
        add_cuts(
            config,
            reg_linear_cuts,
            reg_lmi_cuts,
            constr_svec_coefs[most_violated_constr_index],
            constr_svec_offset[most_violated_constr_index],
            x,
            eval_x.eigenvalues[most_violated_constr_index],
            eval_x.eigenvectors[most_violated_constr_index],
            config.n_linear_cuts_for_regularised_rmp,
            config.n_lmi_cuts_for_regularised_rmp,
        )

        n_reg_linear_cuts = reg_linear_cuts.n
        n_reg_lmi_cuts = reg_lmi_cuts.n
        n_unreg_linear_cuts = unreg_linear_cuts.n
        n_unreg_lmi_cuts = unreg_lmi_cuts.n

        # Do subgradient projection.
        funcval = eval_x.g[most_violated_constr_index]
        vec = eval_x.eigenvectors[most_violated_constr_index][:, 0]
        v0v0t = cpsdppy.linalg.svec(vec[:, None] @ vec[None, :])
        np.testing.assert_equal(v0v0t.ndim, 1)
        subgrad = -v0v0t @ constr_svec_coefs[most_violated_constr_index]
        np.testing.assert_equal(subgrad.ndim, 1)
        relaxation_parameter = 1.0
        if funcval > 0:
            v = (
                x
                - relaxation_parameter
                * funcval
                * subgrad
                / np.linalg.norm(subgrad) ** 2
            )
        else:
            v = x
        v = reg.project(v)

        # Compute the objetive value and constraint violation of v.
        eval_v = evaluate_solution(v)
        if np.max(eval_v.g) <= config.feas_tol:
            ub = min(ub, eval_v.f)
            best_ub = min(ub, best_ub)

        x = reg.project(v - reg.step_size * objective_coef)

        step_size_manager.feed(
            x=x,
            fx=eval_x.f,
            gx=eval_x.g,
            v=v,
            fv=eval_v.f,
            gv=eval_v.g,
        )
        reg.step_size = step_size_manager.step_size

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
            step_size=reg.step_size,
            x=x,
            fx=eval_x.f,
            fx_gap=eval_x.f_gap,
            gx=eval_x.g,
            v=v,
            fv=eval_v.f,
            fv_gap=eval_v.f_gap,
            gv=eval_v.g,
            regularised_rmp_n_linear_cuts=reg_linear_cuts.n,
            regularised_rmp_n_lmi_cuts=reg_lmi_cuts.n,
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
        n_rcuts = n_reg_linear_cuts + n_reg_lmi_cuts
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
            f"  {format_number(eval_v.f_gap * 100, width=11)}",
            f"  {format_number(np.max(eval_v.g), width=8)}",
            # f"  {format_number(_ub_gap * 100, width=11)}",
            # f"  {format_number(lb, width=11)}",
            # lb_symbol,
            f"  {format_number(-_lb_gap * 100, width=11)}",
            f"  {n_rcuts:5d}",
            f"  {n_ucuts:5d}",
            f"  {reg.step_size:7.1e}",
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
        solution_found |= (
            np.isfinite(eval_v.f_gap)
            and (0 < config.tol)
            and (eval_v.f_gap <= config.tol)
            and (np.max(eval_v.g) <= config.feas_tol)
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


class StepSizeManager:
    def __init__(self, config):
        self.config = config
        self.shift = 0
        self.scale = 1.2
        self.step_size_factor = 1
        self.fx = np.array([])
        self.gx = np.array([])
        self.fv = np.array([])
        self.gv = np.array([])

    @property
    def step_size(self):
        return self.config.step_size * self.step_size_factor

    def feed(self, x, fx, gx, v, fv, gv):
        self.fx = np.r_[self.fx, fx]
        self.gx = np.r_[self.gx, np.max(gx)]
        self.fv = np.r_[self.fv, fv]
        self.gv = np.r_[self.gv, np.max(gv)]

        iter = len(self.gx)

        warmup = 4
        v_x = np.linalg.norm(v - x, ord=2)

        if iter <= warmup:
            step_size_adjustament = "none"
        elif v_x <= self.config.feas_tol:
            step_size_adjustament = "increase"
        elif np.all(self.gv[warmup:] > 1e-3):
            step_size_adjustament = "decrease"
        else:
            score = self.fv.copy()
            score[:warmup] = np.inf
            score[self.gv > self.config.feas_tol] = np.inf
            if score[-1] <= np.min(score[:-1]):
                step_size_adjustament = "increase"
            else:
                step_size_adjustament = "decrease"

        if iter >= warmup:
            if step_size_adjustament == "increase":
                self.step_size_factor += self.shift
                self.step_size_factor *= self.scale
            elif step_size_adjustament == "decrease":
                self.step_size_factor -= self.shift
                self.step_size_factor /= self.scale


def run_column_generation(problem_data, config):
    model = cpsdppy.mip_solvers.gurobi.GurobiInterface()
    lb = problem_data["variable_lb"]
    ub = problem_data["variable_ub"]
    objective_coef = problem_data["objective_coefficient"]
    model.add_variables(lb=lb, ub=ub, obj=objective_coef)
    linear_cuts = cpsdppy.mip_solver_extensions.LinearCuts(model)
    lmi_cuts = cpsdppy.mip_solver_extensions.LMICuts(model)
    n_variables = model.get_n_variables()

    constr_coefs = problem_data["lmi_constraint_coefficient"]
    constr_offsets = problem_data["lmi_constraint_offset"]
    constr_svec_coefs = problem_data["lmi_svec_constraint_coefficient"]
    constr_svec_offset = problem_data["lmi_svec_constraint_offset"]

    assert config.initial_cut_type in ["linear", "lmi", "none"]
    for coef_i in range(len(constr_svec_coefs)):
        if config.initial_cut_type == "linear":
            add_initial_linear_cuts(
                linear_cuts, constr_coefs[coef_i], constr_offsets[coef_i]
            )
        elif config.initial_cut_type == "lmi":
            add_initial_lmi_cuts(
                lmi_cuts, constr_coefs[coef_i], constr_offsets[coef_i]
            )

    best_lb = -np.inf

    for iteration in range(config.iteration_limit):
        linear_cuts.iteration = iteration
        lmi_cuts.iteration = iteration
        model.solve()
        lb = model.get_objective_value()
        best_lb = max(lb, best_lb)
        if not model.is_optimal():
            raise ValueError(f"{iteration=}  {model.get_status_name()=}")
        x = model.get_solution()[:n_variables]

        matrices = []
        eigenvectors = []
        eigenvalues = []

        for coef_i in range(len(constr_svec_coefs)):
            matrix = cpsdppy.linalg.svec_inv(
                constr_svec_coefs[coef_i] @ x - constr_svec_offset[coef_i],
                part="f",
            )
            matrices.append(matrix)
            w, v = np.linalg.eigh(matrix)
            eigenvalues.append(w)
            eigenvectors.append(v)

        coef_i = np.argmin([i[0] for i in eigenvalues])

        matrix = matrices[coef_i]
        w = eigenvalues[coef_i]
        v = eigenvectors[coef_i]

        add_cuts(
            config,
            linear_cuts,
            lmi_cuts,
            constr_svec_coefs[coef_i],
            constr_svec_offset[coef_i],
            x,
            w,
            v,
            config.n_linear_cuts_for_unregularised_rmp,
            config.n_lmi_cuts_for_unregularised_rmp,
        )

        obj = objective_coef @ x
        constr = -w[0]
        n_linear_cuts = linear_cuts.n
        n_lmi_cuts = lmi_cuts.n
        if iteration == 0:
            logger.info(
                f"{'it':>3s} "
                f"{'obj':>9s} {'constr':>9s} "
                f"{'|x|inf':>9s} "
                f"{'lnrcuts'} {'lmicuts'}"
            )
        logger.info(
            f"{iteration + 1:3d} "
            f"{utils.format_number(obj)} "
            f"{utils.format_number(constr)} "
            f"{utils.format_number(np.linalg.norm(x, ord=np.inf))} "
            f"{n_linear_cuts:7d} {n_lmi_cuts:7d}"
        )

        _lb_gap = gap(
            best_lb,
            problem_data["target_objective"],
            problem_data["target_objective"],
        )
        if np.isfinite(_lb_gap) and (0 <= -_lb_gap <= config.tol):
            solver_status = "gap_closed"
            break

    return {
        "solver_status": solver_status,
        "linear_cuts": linear_cuts,
        "constr_svec_coefs": constr_svec_coefs,
        "constr_svec_offset": constr_svec_offset,
    }


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
            problem_data = cpsdppy.sdpa.read(problem_name)

            results = dict()

            for setup in iter:
                logger.info("- " * 20)
                logger.info(str(setup))
                logger.info("- " * 20)

                config, prefix = update_config(
                    problem_name, base_config, step_size, setup
                )

                results[prefix] = run_subgradient_projection(
                    prefix, problem_data, config
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

                res = results[prefix]
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
