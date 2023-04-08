# -*- coding: utf-8 -*-

"""Linear and linear matrix inequality cuts"""

import logging
import weakref

import indexremove
import numpy as np
import scipy.sparse

logger = logging.getLogger(__name__)


class LinearCuts:
    """Linear cuts

    Examples
    --------
    >>> import cpsdppy
    >>> m = cpsdppy.mip_solvers.gurobi.GurobiInterface()
    >>> _ = m.add_variables(lb=-2, ub=2, obj=[1, 2])
    >>> linear_cuts = LinearCuts(m)

    [ 1 + x     y   ]
    [               ]  >=  0
    [   y     r - x ]

    >>> coefs = np.array([
    ...     [
    ...         [ 1,  0],
    ...         [ 0, -1],
    ...     ],
    ...     [
    ...         [ 0,  1],
    ...         [ 1,  0],
    ...     ],
    ... ])
    >>> offset = np.array([[1, 0], [0, 1]])

    >>> for i in range(5):
    ...     linear_cuts.iteration = i
    ...     m.solve()
    ...     x = m.get_solution()
    ...     matrix = np.sum(coefs * x[:, None, None], axis=0) + offset
    ...     w, v = np.linalg.eigh(matrix)
    ...     # f = -lambda_min(x a + y b + z c)
    ...     # g_x = -v_min(x a + y b + c) v_min(x a + y b + c)^T bullet a
    ...     #     = -v_min(x a + y b + c) a v_min(x a + y b + c)^T
    ...     f = -w[0]
    ...     v_min = v[:, 0]
    ...     g = -np.array([
    ...         v_min @ coefs[0] @ v_min,
    ...         v_min @ coefs[1] @ v_min,
    ...     ])
    ...     _offset = f - g @ x
    ...     linear_cuts.add_linear_cuts(coef=g, offset=_offset)
    ...     print(f"{f:8.4f}  {x[0]:8.4f}, {x[1]:8.4f}")
    1.8284   -2.0000,  -2.0000
    1.0840    0.5858,  -2.0000
    0.1625   -0.2879,  -1.1263
    0.0369   -0.5132,  -0.9010
    0.0090   -0.3780,  -0.9355
    """

    def __init__(self, model):
        """Initialise a LinearCuts instance"""
        self.linear_constraint_index = np.array([], dtype=int)
        self.last_active_iteration = np.array([], dtype=int)
        self.iteration = 0
        self.model = weakref.ref(model)

        model.add_hooks(self)

    @property
    def n(self):
        return self.linear_constraint_index.size

    def add_linear_cuts(self, coef, offset):
        """Add linear cuts

        This adds cuts of the form
        ```
        coef[i].T x + offset[i] <= 0.
        ```
        """
        model = self.model()
        if isinstance(coef, scipy.sparse.spmatrix):
            n_new_cuts = coef.shape[0]
            new_constraint_index = model.add_linear_constraints(
                sense="L", rhs=-offset
            )
            coef = coef.tocoo()
            offset = np.atleast_1d(offset)
            row = new_constraint_index[coef.row]
            model.set_linear_constraint_coefs(zip(row, coef.col, coef.data))
        else:
            coef = np.atleast_2d(coef)
            offset = np.atleast_1d(offset)
            n_new_cuts, n_vars = coef.shape
            new_constraint_index = model.add_linear_constraints(
                sense="L", rhs=-offset
            )
            row = np.repeat(new_constraint_index, n_vars)
            col = np.tile(np.arange(coef.shape[1]), n_new_cuts)
            model.set_linear_constraint_coefs(zip(row, col, coef.ravel()))
        self.linear_constraint_index = np.concatenate(
            [
                self.linear_constraint_index,
                new_constraint_index,
            ]
        )
        self.last_active_iteration = np.concatenate(
            [self.last_active_iteration, np.full(n_new_cuts, self.iteration)]
        )
        logger.debug(f"{self.__class__.__name__} added {n_new_cuts} cuts")

    def remove_linear_constraints_hook(self, model, index):
        indexremove.remove(self.linear_constraint_index.ravel(), index.ravel())

    def solve_enter_hook(self, model):
        memory = 10
        buf = self.last_active_iteration <= self.iteration - memory
        dropped = np.nonzero(buf)[0]
        kept = np.nonzero(~buf)[0]
        if dropped.size > 0:
            model.remove_linear_constraints(
                self.linear_constraint_index[dropped]
            )
            self.linear_constraint_index = self.linear_constraint_index[kept]
            self.last_active_iteration = self.last_active_iteration[kept]
        logger.debug(f"{self.__class__.__name__} removed {dropped.size} cuts")

    def solve_exit_hook(self, model):
        slacks = model.get_linear_constraint_slacks()[
            self.linear_constraint_index
        ]
        self.last_active_iteration[np.abs(slacks) >= 1e-6] = self.iteration


class LMICuts:
    """Linear matrix inequality cuts

    Examples
    --------
    >>> import cpsdppy
    >>> m = cpsdppy.mip_solvers.gurobi.GurobiInterface()
    >>> _ = m.add_variables(lb=-2, ub=2, obj=[1, 2])
    >>> lmi_cuts = LMICuts(m)

    [ 1 + x     y   ]
    [               ]  >=  0
    [   y     r - x ]

    >>> coefs = np.array([
    ...     [
    ...         [ 1,  0],
    ...         [ 0, -1],
    ...     ],
    ...     [
    ...         [ 0,  1],
    ...         [ 1,  0],
    ...     ],
    ... ])
    >>> offset = np.array([[1, 0], [0, 1]])

    >>> for i in range(3):
    ...     lmi_cuts.iteration = i
    ...     m.solve()
    ...     x = m.get_solution()[:2]
    ...     matrix = np.sum(coefs * x[:, None, None], axis=0) + offset
    ...     w, v = np.linalg.eigh(matrix)
    ...     # v^T (x a + y b + z c) v : PSD
    ...     # x v^T a v + y v^T b v + z v^T c v : PSD
    ...     cut_coef = np.stack([
    ...         [v[:, 0] @ coefs[0] @ v[:, 0], v[:, 0] @ coefs[1] @ v[:, 0]],
    ...         [v[:, 0] @ coefs[0] @ v[:, 1], v[:, 0] @ coefs[1] @ v[:, 1]],
    ...         [v[:, 1] @ coefs[0] @ v[:, 1], v[:, 1] @ coefs[1] @ v[:, 1]],
    ...     ])
    ...     cut_offset = np.array([
    ...         v[:, 0] @ offset @ v[:, 0],
    ...         v[:, 0] @ offset @ v[:, 1],
    ...         v[:, 1] @ offset @ v[:, 1],
    ...     ])
    ...     lmi_cuts.add_lmi_cuts(cut_coef, cut_offset)
    ...     f = w[0]
    ...     print(f"{f:8.4f}  {x[0]:8.4f}, {x[1]:8.4f}")
    -1.8284   -2.0000,  -2.0000
     0.0000   -0.4473,  -0.8944
     0.0000   -0.4470,  -0.8946
    """

    def __init__(self, model):
        """Initialise a LMICuts instance"""
        self.linear_constraint_index = np.array([], dtype=int).reshape(0, 3)
        self.quadratic_constraint_index = np.array([], dtype=int)
        self.variable_index = np.array([], dtype=int).reshape(0, 3)
        self.last_active_iteration = np.array([], dtype=int)
        self.iteration = 0
        self.n = 0
        self.n_variables = model.get_n_variables()
        self.model = weakref.ref(model)

        model.add_hooks(self)

        self.coef = scipy.sparse.csr_matrix(
            ([], ([], [])), shape=(0, model.get_n_variables())
        )
        self.offset = np.array([]).reshape(0, 3)

    def add_lmi_cuts(self, coef, offset):
        """Add LMI cuts

        This adds cuts of the form
        ```
        coef[i] x + offset[i] : PSDCone
        ```
        """
        if coef.ndim == 2:
            coef = coef[None]
        offset = np.atleast_2d(offset)
        n_new_cuts = len(coef)
        self.n += n_new_cuts
        model = self.model()

        for _coef, _offset in zip(coef, offset):
            n = model.get_n_variables()
            Z = np.zeros((3, n - _coef.shape[1]))
            (
                new_variable_index,
                new_quadratic_constraint_index,
            ) = model.add_2x2_psd_variables()
            D = np.array(
                [
                    [
                        -1,
                        0,
                        0,
                    ],
                    [
                        0,
                        -1,
                        0,
                    ],
                    [
                        0,
                        0,
                        -1,
                    ],
                ]
            )
            CZD = np.concatenate([_coef, Z, D], axis=1)
            # coef = scipy.sparse.vstack([_coef, D])
            coef = scipy.sparse.csr_array(CZD)
            new_linear_constraint_index = model.add_linear_constraints(
                shape=3, coef=coef, sense="E", rhs=-_offset
            )

            self.coef = scipy.sparse.vstack([self.coef, _coef]).tocsr()

            self.offset = np.concatenate(
                [self.offset, _offset[None, :]], axis=0
            )

            self.variable_index = np.concatenate(
                [
                    self.variable_index,
                    new_variable_index,
                ],
                axis=0,
            )
            self.linear_constraint_index = np.concatenate(
                [
                    self.linear_constraint_index,
                    new_linear_constraint_index[None],
                ],
                axis=0,
            )
            self.quadratic_constraint_index = np.concatenate(
                [
                    self.quadratic_constraint_index,
                    new_quadratic_constraint_index,
                ]
            )
            self.last_active_iteration = np.concatenate(
                [
                    self.last_active_iteration,
                    np.full(n_new_cuts, self.iteration),
                ]
            )
        logger.debug(f"{self.__class__.__name__} added {n_new_cuts} cuts")

    def remove_variables_hook(self, model, index):
        indexremove.remove(self.variable_index.ravel(), index.ravel())

    def remove_linear_constraints_hook(self, model, index):
        indexremove.remove(self.linear_constraint_index.ravel(), index.ravel())

    def remove_quadratic_constraints_hook(self, model, index):
        indexremove.remove(
            self.quadratic_constraint_index.ravel(), index.ravel()
        )

    def solve_enter_hook(self, model):
        memory = 10
        buf = self.last_active_iteration <= self.iteration - memory
        dropped = np.nonzero(buf)[0]
        kept = np.nonzero(~buf)[0]
        if dropped.size > 0:
            model.remove_linear_constraints(
                self.linear_constraint_index[dropped]
            )
            model.remove_quadratic_constraints(
                self.quadratic_constraint_index[dropped]
            )
            model.remove_variables(self.variable_index[dropped])
            self.linear_constraint_index = self.linear_constraint_index[kept]
            self.quadratic_constraint_index = self.quadratic_constraint_index[
                kept
            ]
            self.variable_index = self.variable_index[kept]
            self.last_active_iteration = self.last_active_iteration[kept]
            self.coef = self.coef[np.repeat(kept, 3)]
            self.offset = self.offset[kept]
            self.n -= dropped.size
        logger.debug(f"{self.__class__.__name__} removed {dropped.size} cuts")

    def solve_exit_hook(self, model):
        slacks = np.empty(self.n)
        x = model.get_solution()[: self.n_variables]
        svec = (self.coef @ x).reshape(-1, 3) + self.offset
        matrix = np.empty((2, 2))
        for i, (a, b, c) in enumerate(svec):
            matrix[0, 0] = a
            matrix[1, 1] = b
            matrix[1, 0] = matrix[0, 1] = c
            w, v = np.linalg.eigh(matrix)
            slacks[i] = w[0]
        self.last_active_iteration[np.abs(slacks) >= 1e-6] = self.iteration


if __name__ == "__main__":
    import doctest

    n_failuers, _ = doctest.testmod(
        optionflags=doctest.ELLIPSIS + doctest.NORMALIZE_WHITESPACE
    )
    if n_failuers > 0:
        raise ValueError(f"failed {n_failuers} tests")

# vimquickrun: python %
