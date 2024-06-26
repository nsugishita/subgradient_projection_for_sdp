# -*- coding: utf-8 -*-

"""Subgradient projection solver"""


import itertools
import logging
import os

import numpy as np

from cpsdppy import linalg, mip_solver_extensions, mip_solvers, utils
from cpsdppy.sdp_solvers import common

logger = logging.getLogger(__name__)


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
        v=dict(default=np.full(n_variables, np.nan), timing=True),
        fv=dict(default=np.nan, timing=True),
        fv_gap=dict(default=np.nan, timing=True),
        gv=dict(default=np.full(len(constr_svec_coefs), np.nan), timing=True),
        regularised_rmp_n_linear_cuts=dict(default=-1, timing=False),
        regularised_rmp_n_lmi_cuts=dict(default=-1, timing=False),
        unregularised_rmp_n_linear_cuts=dict(default=-1, timing=False),
        unregularised_rmp_n_lmi_cuts=dict(default=-1, timing=False),
    )

    regularised_model = mip_solvers.get_solver_interface(
        config.solver_interface
    )
    unregularised_model = mip_solvers.get_solver_interface(
        config.solver_interface
    )
    xlb = problem_data["variable_lb"]
    xub = problem_data["variable_ub"]
    objective_coef = problem_data["objective_coefficient"]
    regularised_model.add_variables(lb=xlb, ub=xub, obj=objective_coef)
    unregularised_model.add_variables(lb=xlb, ub=xub, obj=objective_coef)
    regularised_model.add_linear_constraints(
        coef=problem_data["linear_constraint_coefficient"],
        rhs=problem_data["linear_constraint_rhs"],
        sense=problem_data["linear_constraint_sense"],
    )
    unregularised_model.add_linear_constraints(
        coef=problem_data["linear_constraint_coefficient"],
        rhs=problem_data["linear_constraint_rhs"],
        sense=problem_data["linear_constraint_sense"],
    )

    reg_linear_cuts = mip_solver_extensions.LinearCuts(
        regularised_model, config
    )
    unreg_linear_cuts = mip_solver_extensions.LinearCuts(
        unregularised_model, config
    )
    reg_lmi_cuts = mip_solver_extensions.LMICuts(regularised_model, config)
    unreg_lmi_cuts = mip_solver_extensions.LMICuts(unregularised_model, config)

    if config.n_linear_cuts_for_regularised_rmp >= 0:
        n_reg_linear_cuts = config.n_linear_cuts_for_regularised_rmp
    else:
        n_reg_linear_cuts = config.n_linear_cuts
    if config.n_lmi_cuts_for_regularised_rmp >= 0:
        n_reg_lmi_cuts = config.n_lmi_cuts_for_regularised_rmp
    else:
        n_reg_lmi_cuts = config.n_lmi_cuts
    if config.n_linear_cuts_for_unregularised_rmp >= 0:
        n_unreg_linear_cuts = config.n_linear_cuts_for_unregularised_rmp
    else:
        n_unreg_linear_cuts = config.n_linear_cuts
    if config.n_lmi_cuts_for_unregularised_rmp >= 0:
        n_unreg_lmi_cuts = config.n_lmi_cuts_for_unregularised_rmp
    else:
        n_unreg_lmi_cuts = config.n_lmi_cuts

    assert config.initial_cut_type in ["linear", "lmi", "none"]
    for coef_i in range(len(constr_svec_coefs)):
        if config.initial_cut_type == "linear":
            unreg_linear_cuts.add_linear_cuts(
                *common.get_initial_linear_cuts(
                    constr_coefs[coef_i], constr_offsets[coef_i]
                )
            )
        elif config.initial_cut_type == "lmi":
            unreg_lmi_cuts.add_lmi_cuts(
                *common.get_initial_lmi_cuts(
                    constr_coefs[coef_i], constr_offsets[coef_i]
                )
            )

    best_lb = -np.inf
    best_ub = np.inf

    if "initial_x" in problem_data:
        x = np.broadcast_to(problem_data["initial_x"], (n_variables,))
    else:
        x = np.zeros(n_variables)
    x = mip_solver_extensions.project(regularised_model, x)
    initial_x = x

    solver_status = "unknown"

    journal.start_hook()
    timer = utils.timer()
    step_size_manager = StepSizeManager(config)

    for iteration in itertools.count():
        # Check if we have reached the limits.
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

        #
        # Evaluation of LB
        #

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

        #
        # Optimality Step (Subgradient Step)
        #

        if not config.feasibility_recovery:
            # Compute the objetive value and constraint violation of v.
            eval_x = common.evaluate_solution(x, problem_data)
            if np.max(eval_x.g) <= config.feas_tol:
                ub = min(ub, eval_x.f)
                best_ub = min(ub, best_ub)

            # TODO Add cuts from all the constraints.
            if config.add_cuts_after_optimality_step:
                most_viol_constr_index_x = np.argmax(eval_x.g)
                if config.eval_lb_every > 0:
                    common.add_cuts(
                        config=config,
                        linear_cuts=unreg_linear_cuts,
                        lmi_cuts=unreg_lmi_cuts,
                        constr_svec_coef=constr_svec_coefs[
                            most_viol_constr_index_x
                        ],
                        constr_svec_offset=constr_svec_offset[
                            most_viol_constr_index_x
                        ],
                        x=x,
                        w=eval_x.eigenvalues[most_viol_constr_index_x],
                        v=eval_x.eigenvectors[most_viol_constr_index_x],
                        n_linear_cuts=n_unreg_linear_cuts,
                        n_lmi_cuts=n_unreg_lmi_cuts,
                    )
                common.add_cuts(
                    config=config,
                    linear_cuts=reg_linear_cuts,
                    lmi_cuts=reg_lmi_cuts,
                    constr_svec_coef=constr_svec_coefs[most_viol_constr_index_x],
                    constr_svec_offset=constr_svec_offset[
                        most_viol_constr_index_x
                    ],
                    x=x,
                    w=eval_x.eigenvalues[most_viol_constr_index_x],
                    v=eval_x.eigenvectors[most_viol_constr_index_x],
                    n_linear_cuts=n_reg_linear_cuts,
                    n_lmi_cuts=n_reg_lmi_cuts,
                )

            step_size = step_size_manager.step_size
            v = x - step_size * objective_coef
            if config.projection_after_optimality_step:
                v = mip_solver_extensions.project(regularised_model, v)

        else:
            v = x
            step_size = 0

        #
        # Feasibility step (Subgradient Projection Step)
        #

        eval_v = common.evaluate_solution(v, problem_data)

        if config.feasibility_recovery:
            eval_x = eval_v

        most_viol_constr_index_v = np.argmax(eval_v.g)
        if np.max(eval_v.g) <= config.feas_tol:
            ub = min(ub, eval_v.f)
            best_ub = min(ub, best_ub)

        # TODO Add cuts from all the constraints.
        if config.eval_lb_every > 0:
            common.add_cuts(
                config=config,
                linear_cuts=unreg_linear_cuts,
                lmi_cuts=unreg_lmi_cuts,
                constr_svec_coef=constr_svec_coefs[most_viol_constr_index_v],
                constr_svec_offset=constr_svec_offset[
                    most_viol_constr_index_v
                ],
                x=v,
                w=eval_v.eigenvalues[most_viol_constr_index_v],
                v=eval_v.eigenvectors[most_viol_constr_index_v],
                n_linear_cuts=n_unreg_linear_cuts,
                n_lmi_cuts=n_unreg_lmi_cuts,
            )
        common.add_cuts(
            config=config,
            linear_cuts=reg_linear_cuts,
            lmi_cuts=reg_lmi_cuts,
            constr_svec_coef=constr_svec_coefs[most_viol_constr_index_v],
            constr_svec_offset=constr_svec_offset[most_viol_constr_index_v],
            x=v,
            w=eval_v.eigenvalues[most_viol_constr_index_v],
            v=eval_v.eigenvectors[most_viol_constr_index_v],
            n_linear_cuts=n_reg_linear_cuts,
            n_lmi_cuts=n_reg_lmi_cuts,
        )

        # Do subgradient projection.
        funcval = eval_v.g[most_viol_constr_index_v]
        vec = eval_v.eigenvectors[most_viol_constr_index_v][:, 0]
        v0v0t = linalg.svec(vec[:, None] @ vec[None, :])
        np.testing.assert_equal(v0v0t.ndim, 1)
        subgrad = -v0v0t @ constr_svec_coefs[most_viol_constr_index_v]
        np.testing.assert_equal(subgrad.ndim, 1)
        relaxation_parameter = 1.0
        if funcval > 0:
            x = v - relaxation_parameter * funcval * subgrad / (
                np.linalg.norm(subgrad) ** 2
            )
            if config.projection_after_feasibility_step:
                x = mip_solver_extensions.project(regularised_model, x)
        else:
            x = v

        _lb_gap = common.gap(
            best_lb,
            problem_data.get("target_objective", np.nan),
            problem_data.get("target_objective", np.nan),
        )
        _ub_gap = common.gap(
            best_ub,
            problem_data.get("target_objective", np.nan),
            problem_data.get("target_objective", np.nan),
        )

        journal.set_iteration_items(
            lb=lb,
            best_lb=best_lb,
            lb_gap=_lb_gap,
            ub=ub,
            best_ub=best_ub,
            ub_gap=_ub_gap,
            step_size=step_size,
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
            f"  {'rcuts':>5s}",
            f"  {'ucuts':>5s}",
            f"  {'ss':>7s}",
        ]
        # lb_symbol = " "
        n_rcuts = reg_linear_cuts.n + reg_lmi_cuts.n
        n_ucuts = unreg_linear_cuts.n + unreg_lmi_cuts.n
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
            f"  {step_size_manager.step_size:7.1e}",
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
        if config.feasibility_recovery:
            if np.max(eval_v.g) <= config.feas_tol:
                solver_status = "solved"
                break
        elif config.termination_criteria == "lb_and_solution":
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

        # Adjust the step size.
        step_size_manager.feed(
            x=x,
            fx=eval_x.f,
            gx=eval_x.g,
            v=v,
            fv=eval_v.f,
            gv=eval_v.g,
        )

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
            initial_x=initial_x,
        )
    )
    journal.dump_data(out=result)
    timer.dump_data(out=result)
    reg_linear_cuts.dump_data(out=result)
    for key in result:
        result[key] = np.asarray(result[key])
    return result


class StepSizeManager:
    def __init__(self, config):
        self.config = config
        self.step_size_factor = 1.0
        self.fx = np.array([])
        self.gx = np.array([])
        self.fv = np.array([])
        self.gv = np.array([])

        self.best_score = np.inf

    @property
    def step_size(self):
        return self.config.step_size * self.step_size_factor

    def feed(self, x, fx, gx, v, fv, gv):
        shift = self.config.step_size_manager_shift
        scale = self.config.step_size_manager_scale

        self.fx = np.r_[self.fx, fx]
        self.gx = np.r_[self.gx, np.max(gx)]
        self.fv = np.r_[self.fv, fv]
        self.gv = np.r_[self.gv, np.max(gv)]

        iter = len(self.gx) - 1

        if self.config.step_size_manager_version == 1:

            warmup = 4
            v_x = np.linalg.norm(v - x, ord=2)

            if iter <= warmup:
                step_size_adjustament = "none"
            elif v_x <= self.config.feas_tol:
                step_size_adjustament = "increase"
            elif np.all(self.gv[warmup:] > 1e-3):
                step_size_adjustament = "decrease"
            else:
                score = self.fx.copy()
                score[:warmup] = np.inf
                score[self.gx > self.config.feas_tol] = np.inf
                if score[-1] <= np.min(score[:-1]):
                    step_size_adjustament = "increase"
                else:
                    step_size_adjustament = "decrease"

            if iter >= warmup:
                if step_size_adjustament == "increase":
                    self.step_size_factor += shift
                    self.step_size_factor *= scale
                elif step_size_adjustament == "decrease":
                    self.step_size_factor -= shift
                    self.step_size_factor /= scale

        elif self.config.step_size_manager_version == 2:
            if iter <= 4:
                return

            score = fv

            if np.max(gv) > self.config.feas_tol:
                step_size_adjustament = "decrease"
            elif self.best_score > score:
                step_size_adjustament = "increase"
                self.best_score = score
            else:
                step_size_adjustament = "decrease"

            if step_size_adjustament == "increase":
                self.step_size_factor += shift
                self.step_size_factor *= scale
            elif step_size_adjustament == "decrease":
                self.step_size_factor -= shift
                self.step_size_factor *= 1 - (scale - 1)

            self.step_size_factor = np.clip(self.step_size_factor, 1e-3, 1e3)


# vimquickrun: . ./scripts/activate.sh ; python %
