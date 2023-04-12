# -*- coding: utf-8 -*-

"""Solve SDPA using column generation"""

# TODO Use config to set memory.
# TODO Log time etc.

import logging

import numpy as np
import scipy.sparse

import cpsdppy

logger = logging.getLogger(__name__)


def add_initial_cuts(lmi_cuts, constr_coef, constr_offset):
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
        lmi_cuts,
        constr_coef,
        constr_offset,
        v0,
        v1,
    )
    lmi_cuts.add_lmi_cuts(coef=coef, offset=offset)


def run_subgradient_projection(problem_data, config):
    model = cpsdppy.mip_solvers.gurobi.GurobiInterface()
    lb = problem_data["variable_lb"]
    ub = problem_data["variable_ub"]
    objective_coef = problem_data["objective_coefficient"]
    model.add_variables(lb=lb, ub=ub, obj=objective_coef)

    reg = cpsdppy.mip_solver_extensions.MoreuYoshidaRegularisation(
        model, config=None
    )
    reg.step_size = 100  # TODO
    linear_cuts = cpsdppy.mip_solver_extensions.LinearCuts(model)
    lmi_cuts = cpsdppy.mip_solver_extensions.LMICuts(model)
    n_variables = model.get_n_variables()

    constr_coefs = problem_data["lmi_constraint_coefficient"]
    constr_offsets = problem_data["lmi_constraint_offset"]
    constr_svec_coefs = problem_data["lmi_svec_constraint_coefficient"]
    constr_svec_offset = problem_data["lmi_svec_constraint_offset"]

    for coef_i in range(len(constr_svec_coefs)):
        add_initial_cuts(
            lmi_cuts, constr_coefs[coef_i], constr_offsets[coef_i]
        )

    x_list = []

    x = np.zeros(n_variables)

    for iteration in range(20):
        linear_cuts.iteration = iteration
        lmi_cuts.iteration = iteration

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

        x = reg.project(x)

        x = reg.prox(x)

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

        if constr <= 1e-6:
            break

    return {
        "x_list": x_list,
        "linear_cuts": linear_cuts,
        "constr_svec_coefs": constr_svec_coefs,
        "constr_svec_offset": constr_svec_offset,
    }


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

    for coef_i in range(len(constr_svec_coefs)):
        add_initial_cuts(
            lmi_cuts, constr_coefs[coef_i], constr_offsets[coef_i]
        )

    x_list = []

    for iteration in range(100):
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

        coef_i = np.argmin([i[0] for i in eigenvalues])

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
    # TODO Improve efficiency using initialisation routine.

    n_linear_cuts = 1
    n_lmi_cuts = 1

    for i in range(n_linear_cuts):
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


def get_lmi_cut_coef(lmi_cuts, constr_coef, constr_offset, v0, v1):
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

        col0 = np.repeat(col0, n1)
        val0 = np.repeat(val0, n1)
        col1 = np.tile(col1, n0)
        val1 = np.tile(val1, n0)

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


def main():
    """Run the main routine of this script"""
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    problem_data = cpsdppy.sdpa.read("theta1.dat-s")
    # problem_data = cpsdppy.sdpa.read("control1.dat-s")
    # problem_data = cpsdppy.toy.get("a")
    # problem_data = cpsdppy.toy.get("b")
    # problem_data = cpsdppy.toy.get("d")
    config = {}
    # run_column_generation(problem_data, config)
    run_subgradient_projection(problem_data, config)


if __name__ == "__main__":
    main()

# vimquickrun: . ./scripts/activate.sh ; python %
