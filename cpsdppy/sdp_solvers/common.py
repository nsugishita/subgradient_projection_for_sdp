# -*- coding: utf-8 -*-

"""Common routines to solve SDP"""

import collections
import logging

import numpy as np
import scipy.sparse

import cpsdppy

logger = logging.getLogger(__name__)

use_cache = True


def gap(a: float, b: float, c: float) -> float:
    if np.all(np.isfinite([a, b, c])):
        return (a - b) / np.abs(c)
    else:
        return np.nan


def remaining_time(config, timer):
    if config.time_limit is None:
        return np.inf
    if config.time_limit <= 0:
        return np.inf
    if not np.isfinite(config.time_limit):
        return np.inf
    return config.time_limit - timer.walltime


def evaluate_solution(x, problem_data):
    """
    >>> import cpsdppy
    >>> problem_data = cpsdppy.toy.get('a')
    >>> x = np.zeros(len(problem_data['objective_coefficient']))
    >>> res = evaluate_solution(x, problem_data)
    >>> res.f
    0.0
    >>> print(res.g)
    [-1.]
    >>> print(res.matrices[0])
    [[1. 0.]
     [0. 1.]]
    >>> print(res.eigenvalues[0])
    [1. 1.]
    """
    objective_coef = problem_data["objective_coefficient"]
    constr_svec_coefs = problem_data["lmi_svec_constraint_coefficient"]
    constr_svec_offset = problem_data["lmi_svec_constraint_offset"]
    target_objective = problem_data.get("target_objective", np.nan)
    f = objective_coef @ x
    f_gap = gap(
        f,
        target_objective,
        target_objective,
    )
    matrices = []
    eigenvalues = []
    eigenvectors = []
    for coef_i in range(len(constr_svec_coefs)):
        matrix = cpsdppy.linalg.svec_inv(
            constr_svec_coefs[coef_i] @ x - constr_svec_offset[coef_i],
            part="f",
        )
        matrices.append(matrix)
        _eigenvalues, _eigenvectors = np.linalg.eigh(matrix)
        eigenvalues.append(_eigenvalues)
        eigenvectors.append(_eigenvectors)
    g = np.array([-_eigenvalues[0] for _eigenvalues in eigenvalues])
    return eval_res_t(f, f_gap, g, matrices, eigenvalues, eigenvectors)


eval_res_t = collections.namedtuple(
    "eval_res_t", "f f_gap g matrices eigenvalues eigenvectors"
)


def get_initial_linear_cuts(constr_coef, constr_offset):
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
    return coef, offset


def get_initial_lmi_cuts(constr_coef, constr_offset):
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
    return coef, offset


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
        if config.lmi_cuts_from_unique_vectors:
            v0 = v[:, 2 * i]
            v1 = v[:, 2 * i + 1]
        else:
            v0 = v[:, i]
            v1 = v[:, i + 1]
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
        if n_lmi_cuts == 0:
            v0 = v[:, i]
        else:
            if config.lmi_cuts_from_unique_vectors:
                v0 = v[:, i + 2 * n_lmi_cuts]
            else:
                v0 = v[:, i + n_lmi_cuts + 1]
        v0v0t = cpsdppy.linalg.svec(v0[:, None] @ v0[None, :])
        cut_coef = v0v0t @ constr_svec_coef
        cut_offset = v0v0t @ constr_svec_offset
        linear_cuts.add_linear_cuts(coef=-cut_coef, offset=-cut_offset)

    if config.eigen_comb_cut:
        negative_matrix = -(v * w.clip(None, 0)) @ v.T
        # negative_matrix is PSD. Thus,
        #   negative_matrix * (sum A x - rhs) >= 0,
        # or equivalently
        #   -sum ((negative_matrix * A) x) <= -negative_matrix * rhs
        norm = (-w).max()
        if norm >= 1e-3:
            negative_matrix = negative_matrix / norm
            negative_vec = cpsdppy.linalg.svec(negative_matrix)
            subgrad = (negative_vec[None, :] @ constr_svec_coef).ravel()
            rhs = negative_vec.dot(constr_svec_offset)
            linear_cuts.add_linear_cuts(coef=-subgrad, offset=-rhs)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# vimquickrun: . ./scripts/activate.sh ; python %
