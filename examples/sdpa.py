# -*- coding: utf-8 -*-

"""Solve SDPA using column generation"""

# TODO Improve addition of initial cuts.
# TODO Use config to set memory.
# TODO Log time etc.
# TODO Implement subgradient projection.

import logging

import numpy as np
import scipy.sparse

import cpsdppy

logger = logging.getLogger(__name__)


def add_initial_cuts(lmi_cuts, constr_svec_coef, constr_svec_offset):
    mat_size = cpsdppy.linalg.from_svec_size_to_original_size(
        constr_svec_offset.size
    )
    v0 = np.zeros(mat_size)
    v1 = np.zeros(mat_size)
    coef_list = []
    offset_list = []
    for i in range(mat_size):
        for j in range(i + 1, mat_size):
            row = [i]
            col = [0]
            val = [1]
            v0 = scipy.sparse.coo_array((val, (row, col)), shape=(mat_size, 1))
            row = [j]
            col = [0]
            val = [1]
            v1 = scipy.sparse.coo_array((val, (row, col)), shape=(mat_size, 1))
            coef, offset = get_lmi_cut_coef(
                lmi_cuts,
                constr_svec_coef,
                constr_svec_offset,
                v0,
                v1,
            )
            coef_list.append(coef)
            offset_list.append(offset)
            coef, offset = get_lmi_cut_coef(
                lmi_cuts,
                constr_svec_coef,
                constr_svec_offset,
                v0,
                -v1,
            )
            coef_list.append(coef)
            offset_list.append(offset)
    offset_list = np.array(offset_list)
    lmi_cuts.add_lmi_cuts(coef=coef_list, offset=offset_list)


def run_column_generation(problem_data, config):
    model = cpsdppy.mip_solvers.gurobi.GurobiInterface()
    lb = problem_data["variable_lb"]
    ub = problem_data["variable_ub"]
    objective_coef = problem_data["objective_coefficient"]
    model.add_variables(lb=lb, ub=ub, obj=objective_coef)
    linear_cuts = cpsdppy.mip_solver_extensions.LinearCuts(model)
    lmi_cuts = cpsdppy.mip_solver_extensions.LMICuts(model)
    n_variables = model.get_n_variables()

    constr_svec_coefs = problem_data["lmi_svec_constraint_coefficient"]
    constr_svec_offset = problem_data["lmi_svec_constraint_offset"]

    for coef_i in range(len(constr_svec_coefs)):
        add_initial_cuts(
            lmi_cuts, constr_svec_coefs[coef_i], constr_svec_offset[coef_i]
        )

    x_list = []

    for iteration in range(1000):
        linear_cuts.iteration = iteration
        lmi_cuts.iteration = iteration
        model.solve()
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

        coef_i = np.argmin([x[0] for x in eigenvalues])

        matrix = matrices[coef_i]
        w = eigenvalues[coef_i]
        v = eigenvectors[coef_i]

        add_cuts(
            linear_cuts,
            lmi_cuts,
            constr_svec_coefs[coef_i],
            constr_svec_offset[coef_i],
            x,
            w,
            v,
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
            f"{cpsdppy.utils.format_number(obj)} "
            f"{cpsdppy.utils.format_number(constr)} "
            f"{cpsdppy.utils.format_number(np.linalg.norm(x, ord=np.inf))} "
            f"{n_linear_cuts:7d} {n_lmi_cuts:7d}"
        )
        x_list.append(x)

        if constr <= 1e-6:
            break

    return {
        "x_list": x_list,
        "linear_cuts": linear_cuts,
        "constr_svec_coefs": constr_svec_coefs,
        "constr_svec_offset": constr_svec_offset,
    }


def add_cuts(
    linear_cuts, lmi_cuts, constr_svec_coef, constr_svec_offset, x, w, v
):
    f = -w[0]
    v0 = v[:, 0]
    v1 = v[:, 1]
    v0v0t = cpsdppy.linalg.svec(v0[:, None] @ v0[None, :])
    # v0v1t = cpsdppy.linalg.svec(v0[:, None] @ v1[None, :])
    v0v1t = (
        cpsdppy.linalg.svec(
            v0[:, None] @ v1[None, :] + v1[:, None] @ v0[None, :]
        )
        / 2
    )
    v1v1t = cpsdppy.linalg.svec(v1[:, None] @ v1[None, :])

    g = -v0v0t @ constr_svec_coef

    _offset = -f + g @ x
    linear_cuts.add_linear_cuts(coef=g, offset=_offset)

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


def get_lmi_cut_coef(lmi_cuts, constr_svec_coef, constr_svec_offset, v0, v1):
    if isinstance(v0, np.ndarray):
        v0 = v0.ravel()[:, None]
    if isinstance(v1, np.ndarray):
        v1 = v1.ravel()[:, None]

    if isinstance(v0, scipy.sparse.spmatrix):
        sparse = True
    else:
        sparse = False

    v0v0t = cpsdppy.linalg.svec(v0 @ v0.T)
    # v0v1t = cpsdppy.linalg.svec(v0[:, None] @ v1[None, :])
    v0v1t = cpsdppy.linalg.svec(v0 @ v1.T + v1 @ v0.T) / 2
    v1v1t = cpsdppy.linalg.svec(v1 @ v1.T)

    if sparse:
        v0v0t = v0v0t.T
        v0v1t = v0v1t.T
        v1v1t = v1v1t.T

    cut_coef = [
        v0v0t @ constr_svec_coef,
        v0v1t @ constr_svec_coef,
        v1v1t @ constr_svec_coef,
    ]

    if isinstance(cut_coef[0], scipy.sparse.spmatrix):
        stack_func = scipy.sparse.vstack
    else:
        stack_func = np.stack

    cut_coef = stack_func(cut_coef)
    if sparse:
        cut_offset = np.array(
            [
                (v0v0t @ constr_svec_offset).item(),
                (v0v1t @ constr_svec_offset).item(),
                (v1v1t @ constr_svec_offset).item(),
            ]
        )
    else:
        cut_offset = np.array(
            [
                v0v0t @ constr_svec_offset,
                v0v1t @ constr_svec_offset,
                v1v1t @ constr_svec_offset,
            ]
        )

    # lmi_cuts.add_lmi_cuts(coef=cut_coef, offset=cut_offset)
    return cut_coef, cut_offset


def main():
    """Run the main routine of this script"""
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    problem_data = cpsdppy.sdpa.read("theta1.dat-s")
    # problem_data = cpsdppy.sdpa.read("control1.dat-s")
    # problem_data = get_problem_data("a")
    # problem_data = get_problem_data("b")
    # problem_data = cpsdppy.toy.get("d")
    config = {}
    run_column_generation(problem_data, config)


if __name__ == "__main__":
    main()

# vimquickrun: . ./scripts/activate.sh ; python %
