# -*- coding: utf-8 -*-

"""Linear matrix inequality cuts"""

import logging
import weakref

import indexremove
import numpy as np
import scipy.sparse

logger = logging.getLogger(__name__)


class LMICuts:
    """Linear matrix inequality cuts

    Examples
    --------
    >>> import cpsdppy
    >>> config = cpsdppy.config.Config()
    >>> m = cpsdppy.mip_solvers.gurobi.GurobiInterface()
    >>> _ = m.add_variables(lb=-2, ub=2, obj=[1, 2])
    >>> lmi_cuts = LMICuts(m, config)

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
    >>> offset = np.array([[-1, 0], [0, -1]])

    >>> for i in range(3):
    ...     lmi_cuts.iteration = i
    ...     m.solve()
    ...     x = m.get_solution()[:2]
    ...     matrix = np.sum(coefs * x[:, None, None], axis=0) - offset
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

    def __init__(self, model, config):
        """Initialise a LMICuts instance"""
        self.linear_constraint_index = np.array([], dtype=int).reshape(0, 3)
        self.quadratic_constraint_index = np.array([], dtype=int)
        self.variable_index = np.array([], dtype=int).reshape(0, 3)
        self.added_iteration = np.array([], dtype=int)
        self.last_active_iteration = np.array([], dtype=int)
        self.iteration = 0
        self.n = 0
        self.n_variables = model.get_n_variables()
        self.model = weakref.ref(model)
        self.config = config

        model.add_hooks(self)

        self.coef = scipy.sparse.csr_matrix(
            ([], ([], [])), shape=(0, model.get_n_variables())
        )
        self.offset = np.array([]).reshape(0, 3)

    def add_lmi_cuts(self, coef, offset):
        """Add LMI cuts

        This adds cuts of the form
        ```
        coef[i] x - offset[i] : PSDCone
        ```
        """
        if isinstance(coef, (list, tuple)):
            if len(coef) == 0:
                return
            if isinstance(coef[0], np.ndarray):
                coef = np.stack(coef)
        elif isinstance(coef, scipy.sparse.spmatrix):
            coef = [coef]
        # coef is a numpy array or a list of sparse matrices
        if isinstance(coef, np.ndarray):
            coef = coef.reshape(-1, self.n_variables)
            coef = scipy.sparse.csr_array(coef)
        else:
            coef = scipy.sparse.vstack(coef)
        # coef is a sparse matrix of shape (3 * n_new_cuts, n_variables)
        n_new_cuts = coef.shape[0] // 3
        offset = offset.reshape(n_new_cuts, 3)
        np.testing.assert_equal(coef.shape, (3 * n_new_cuts, self.n_variables))
        self.n += n_new_cuts
        model = self.model()
        n = model.get_n_variables()
        Z = scipy.sparse.csr_array(
            ([], ([], [])), shape=(3 * n_new_cuts, n - coef.shape[1])
        )
        (
            new_variable_index,
            new_quadratic_constraint_index,
        ) = model.add_2x2_psd_variables(n_new_cuts)
        D = -scipy.sparse.eye(3 * n_new_cuts)
        constr_coef = scipy.sparse.hstack([coef, Z, D])
        new_linear_constraint_index = model.add_linear_constraints(
            shape=3 * n_new_cuts,
            coef=constr_coef,
            sense="E",
            rhs=offset.ravel(),
        )
        new_linear_constraint_index = new_linear_constraint_index.reshape(
            n_new_cuts, 3
        )

        self.coef = scipy.sparse.vstack([self.coef, coef]).tocsr()

        self.offset = np.concatenate([self.offset, offset], axis=0)

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
                new_linear_constraint_index,
            ],
            axis=0,
        )
        self.quadratic_constraint_index = np.concatenate(
            [
                self.quadratic_constraint_index,
                new_quadratic_constraint_index,
            ]
        )

        self.added_iteration = np.concatenate(
            [
                self.added_iteration,
                np.full(n_new_cuts, self.iteration),
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
        memory = self.config.memory
        if memory < 0:
            memory = np.inf
        if self.config.cut_deletion_criterion == "activity":
            cut_iteration = self.last_active_iteration
        elif self.config.cut_deletion_criterion == "creation":
            cut_iteration = self.added_iteration
        else:
            raise ValueError(
                f"unknown criterion {self.config.cut_deletion_criterion}"
            )
        buf = cut_iteration <= self.iteration - memory
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
            # TODO When cut_coef_unique_list is implemented, drop values here.
            # self.cut_coef_unique_list.erase(dropped)
            self.coef = self.coef[np.repeat(kept, 3)]
            self.offset = self.offset[kept]
            self.n -= dropped.size
        logger.debug(f"{self.__class__.__name__} removed {dropped.size} cuts")

    def solve_exit_hook(self, model):
        model.assert_optimal(suboptimal=True)
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
        self.last_active_iteration[slacks <= 1e-6] = self.iteration


if __name__ == "__main__":
    import doctest

    n_failuers, _ = doctest.testmod(
        optionflags=doctest.ELLIPSIS + doctest.NORMALIZE_WHITESPACE
    )
    if n_failuers > 0:
        raise ValueError(f"failed {n_failuers} tests")

# vimquickrun: python %
