# -*- coding: utf-8 -*-

"""Linear matrix inequality cuts"""

import logging
import weakref

import indexremove
import numpy as np
import scipy.sparse
import uniquelist

logger = logging.getLogger(__name__)

# TODO Improve dupliate cut check in LMICuts
# Currently we naively compare the coef. But in LMICuts
# different coefficient may result in the same constraint.


class LMICuts:
    """Linear matrix inequality cuts

    Examples
    --------
    >>> import cpsdppy
    >>> config = cpsdppy.config.Config()
    >>> m = cpsdppy.mip_solvers.gurobi_interface.GurobiInterface()
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

        self.sparse_coef = False

        if self.config.duplicate_cut_check:
            self.sparse_coef = False
            self.cut_coef_unique_list = uniquelist.UniqueArrayList(
                model.get_n_variables()
            )

        if self.sparse_coef:
            self.cut_coef = scipy.sparse.csr_matrix(
                ([], ([], [])), shape=(0, model.get_n_variables())
            )
        else:
            self.cut_coef = np.array([], dtype=float).reshape(
                0, model.get_n_variables()
            )
        self.cut_offset = np.array([]).reshape(0, 3)

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

        if self.sparse_coef:
            if isinstance(coef, np.ndarray):
                coef = coef.reshape(-1, self.n_variables)
                coef = scipy.sparse.csr_array(coef)
            else:
                coef = scipy.sparse.vstack(coef)
            sparse_coef = coef
        else:
            # coef is a numpy array or a list of sparse matrices
            if not isinstance(coef, np.ndarray):
                coef = scipy.sparse.vstack(coef).toarray()
            if isinstance(coef, np.ndarray):
                coef = coef.reshape(-1, self.n_variables)
            sparse_coef = scipy.sparse.coo_array(coef)

        if self.config.duplicate_cut_check:
            coef = coef.reshape(-1, 3 * self.n_variables)
            pos_new = np.array(
                [self.cut_coef_unique_list.push_back(x) for x in coef]
            )
            pos = np.array([x[0] for x in pos_new])
            new = np.array([x[1] for x in pos_new]).astype(bool)
            self.added_iteration[pos[~new]] = self.iteration
            if np.all(~new):
                return
            coef = coef[new]
            coef = coef.reshape(-1, self.n_variables)

        # coef is a sparse/numpy array of shape (3 * n_new_cuts, n_variables)
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
        constr_coef = scipy.sparse.hstack([sparse_coef, Z, D])
        new_linear_constraint_index = model.add_linear_constraints(
            shape=3 * n_new_cuts,
            coef=constr_coef,
            sense="E",
            rhs=offset.ravel(),
        )
        new_linear_constraint_index = new_linear_constraint_index.reshape(
            n_new_cuts, 3
        )

        if self.sparse_coef:
            self.cut_coef = scipy.sparse.vstack([self.cut_coef, coef]).tocsr()
        else:
            self.cut_coef = np.concatenate([self.cut_coef, coef], axis=0)

        self.cut_offset = np.concatenate([self.cut_offset, offset], axis=0)

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
            if self.config.duplicate_cut_check:
                self.cut_coef_unique_list.erase(dropped)
            self.cut_coef = self.cut_coef[np.repeat(kept, 3)]
            self.cut_offset = self.cut_offset[kept]
            self.n -= dropped.size
        logger.debug(f"{self.__class__.__name__} removed {dropped.size} cuts")

    def solve_exit_hook(self, model):
        model.assert_optimal(suboptimal=True)
        slacks = np.empty(self.n)
        x = model.get_solution()[: self.n_variables]
        svec = (self.cut_coef @ x).reshape(-1, 3) + self.cut_offset
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
