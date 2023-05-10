# -*- coding: utf-8 -*-

"""Cutting-plane solver of SDP"""

import logging
import os

import numpy as np

from cpsdppy import mip_solver_extensions, mip_solvers, utils
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
        regularised_rmp_n_linear_cuts=dict(default=0, timing=False),
        regularised_rmp_n_lmi_cuts=dict(default=0, timing=False),
        unregularised_rmp_n_linear_cuts=dict(default=-1, timing=False),
        unregularised_rmp_n_lmi_cuts=dict(default=-1, timing=False),
    )

    unregularised_model = mip_solvers.get_solver_interface(
        config.solver_interface
    )
    xlb = problem_data["variable_lb"]
    xub = problem_data["variable_ub"]
    objective_coef = problem_data["objective_coefficient"]
    unregularised_model.add_variables(lb=xlb, ub=xub, obj=objective_coef)

    unreg_linear_cuts = mip_solver_extensions.LinearCuts(unregularised_model)
    unreg_lmi_cuts = mip_solver_extensions.LMICuts(unregularised_model)

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

    solver_status = "unknown"

    journal.start_hook()
    timer = utils.timer()

    for iteration in range(10000):
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

        eval_x = common.evaluate_solution(x, problem_data)
        most_violated_constr_index = np.argmax(eval_x.g)
        if np.max(eval_x.g) <= config.feas_tol:
            ub = min(ub, eval_x.f)
            best_ub = min(ub, best_ub)

        common.add_cuts(
            config,
            unreg_linear_cuts,
            unreg_lmi_cuts,
            constr_svec_coefs[most_violated_constr_index],
            constr_svec_offset[most_violated_constr_index],
            x,
            eval_x.eigenvalues[most_violated_constr_index],
            eval_x.eigenvectors[most_violated_constr_index],
            n_unreg_linear_cuts,
            n_unreg_lmi_cuts,
        )

        _lb_gap = common.gap(
            best_lb,
            problem_data["target_objective"],
            problem_data["target_objective"],
        )
        _ub_gap = common.gap(
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


# vimquickrun: . ./scripts/activate.sh ; python %
