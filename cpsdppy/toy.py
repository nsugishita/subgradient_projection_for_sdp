# -*- coding: utf-8 -*-

"""Get toy problem data"""

import numpy as np
import scipy.sparse

from cpsdppy import linalg


def get(problem_name, config=None):
    """Get toy problem data

    Examples
    --------
    >>> import cpsdppy
    >>> data = cpsdppy.toy.get('a')
    >>> print(data['objective_coefficient'])
    [0. 1.]
    """
    n_variables = 2
    if problem_name == "a":
        objective_coef = np.array([0.0, 1.0])

        a = np.array([[1, 0], [0, -1]], dtype=float)
        b = np.array([[0, 1], [1, 0]], dtype=float)
        c = -np.array([[1, 0], [0, 1]], dtype=float)

    elif problem_name == "b":
        objective_coef = np.array([0.0, 1.0])

        a = np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=float,
        )
        b = np.array(
            [
                [0, 0, 0.8],
                [0, 0, 0],
                [0.8, 0, 0],
            ],
            dtype=float,
        )
        b = np.array(
            [
                [0, 0, 1],
                [0, 0, 0],
                [1, 0, 0],
            ],
            dtype=float,
        )
        c = -np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )

    elif problem_name == "c":
        objective_coef = np.array([-1, 0])
        a = np.array(
            [
                [0, 0, 1, 0],
                [0, 0, -1, 1],
                [1, -1, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=float,
        )
        b = np.array(
            [
                [0, 1, 0, -1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [-1, 0, 1, 0],
            ],
            dtype=float,
        )
        c = -np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

    elif problem_name == "d":
        objective_coef = np.array([0.0, 1.0])
        a = np.array(
            [
                [0, 0, 1, 0],
                [0, 0, -0.5, 1],
                [1, -0.5, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=float,
        )
        b = np.array(
            [
                [0, 0.5, 0, -1],
                [0.5, 0, 0.5, 0],
                [0, 0.5, 0, 0.5],
                [-1, 0, 0.5, 0],
            ],
            dtype=float,
        )
        c = -np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

    constr_coefs = np.stack([a, b])
    constr_offset = c
    constr_svec_coefs = np.stack(
        [linalg.svec(x) for x in constr_coefs], axis=1
    )
    constr_svec_offset = linalg.svec(constr_offset)

    linear_constraint_coefficient = scipy.sparse.csr_array(
        ([], ([], [])), shape=(0, n_variables)
    )
    linear_constraint_sense = np.array([], dtype=int)
    linear_constraint_rhs = np.array([], dtype=float)

    return dict(
        n_variables=n_variables,
        objective_sense="min",
        objective_offset=0.0,
        variable_lb=np.array([-2, -2]),
        variable_ub=np.array([2, 2]),
        objective_coefficient=objective_coef,
        linear_constraint_coefficient=linear_constraint_coefficient,
        linear_constraint_sense=linear_constraint_sense,
        linear_constraint_rhs=linear_constraint_rhs,
        lmi_constraint_coefficient=[constr_coefs],
        lmi_constraint_offset=[constr_offset],
        lmi_svec_constraint_coefficient=[constr_svec_coefs],
        lmi_svec_constraint_offset=[constr_svec_offset],
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
