# -*- coding: utf-8 -*-

"""Linear cuts"""

import logging
import weakref

import indexremove
import numpy as np
import scipy.sparse
import uniquelist

logger = logging.getLogger(__name__)


class LinearCuts:
    """Linear cuts

    Examples
    --------
    >>> import cpsdppy
    >>> config = cpsdppy.config.Config()
    >>> m = cpsdppy.mip_solvers.gurobi.GurobiInterface()
    >>> _ = m.add_variables(lb=-2, ub=2, obj=[1, 2])
    >>> linear_cuts = LinearCuts(m, config)

    [ 1 + x     y   ]
    [               ]  >=  0
    [   y     1 - x ]

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

    >>> for i in range(5):
    ...     linear_cuts.iteration = i
    ...     m.solve()
    ...     x = m.get_solution()
    ...     matrix = np.sum(coefs * x[:, None, None], axis=0) - offset
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
    ...     _offset = -f + g @ x
    ...     linear_cuts.add_linear_cuts(coef=g, offset=_offset)
    ...     print(f"{f:8.4f}  {x[0]:8.4f}, {x[1]:8.4f}")
    1.8284   -2.0000,  -2.0000
    1.0840    0.5858,  -2.0000
    0.1625   -0.2879,  -1.1263
    0.0369   -0.5132,  -0.9010
    0.0090   -0.3780,  -0.9355
    """

    def __init__(self, model, config):
        """Initialise a LinearCuts instance"""
        self.linear_constraint_index = np.array([], dtype=int)
        self.added_iteration = np.array([], dtype=int)
        self.last_active_iteration = np.array([], dtype=int)
        self.iteration = 0
        self.model = weakref.ref(model)
        self.config = config

        model.add_hooks(self)

        self.coef = np.array([], dtype=float).reshape(
            0, model.get_n_variables()
        )
        self.offset = np.array([], dtype=float)

        if self.config.duplicate_cut_check:
            self.cut_coef_unique_list = uniquelist.UniqueArrayList(
                model.get_n_variables()
            )

    @property
    def n(self):
        return self.linear_constraint_index.size

    def add_linear_cuts(self, coef, offset):
        """Add linear cuts

        This adds cuts of the form
        ```
        coef[i].T x - offset[i] <= 0.
        ```
        """
        model = self.model()

        if self.config.duplicate_cut_check:
            if isinstance(coef, scipy.sparse.spmatrix):
                coef = coef.toarray()
            coef = np.atleast_2d(coef)
            pos_new = np.array(
                [self.cut_coef_unique_list.push_back(x) for x in coef]
            )
            pos = np.array([x[0] for x in pos_new])
            new = np.array([x[1] for x in pos_new]).astype(bool)
            self.added_iteration[pos[~new]] = self.iteration
            if np.all(~new):
                return
            coef = coef[new]

        if isinstance(coef, scipy.sparse.spmatrix):
            n_new_cuts = coef.shape[0]
            new_constraint_index = model.add_linear_constraints(
                sense="L", rhs=offset
            )
            coef = coef.tocoo()
            offset = np.atleast_1d(offset)
            row = new_constraint_index[coef.row]
            model.set_linear_constraint_coefs(zip(row, coef.col, coef.data))

            self.coef = np.concatenate([self.coef, coef.toarray()])
            self.offset = np.concatenate([self.offset, offset])
        else:
            coef = np.atleast_2d(coef)
            offset = np.atleast_1d(offset)
            n_new_cuts, n_vars = coef.shape
            new_constraint_index = model.add_linear_constraints(
                sense="L", rhs=offset
            )
            row = np.repeat(new_constraint_index, n_vars)
            col = np.tile(np.arange(coef.shape[1]), n_new_cuts)
            model.set_linear_constraint_coefs(zip(row, col, coef.ravel()))

            self.coef = np.concatenate([self.coef, coef])
            self.offset = np.concatenate([self.offset, offset])
        self.linear_constraint_index = np.concatenate(
            [
                self.linear_constraint_index,
                new_constraint_index,
            ]
        )
        self.added_iteration = np.concatenate(
            [self.added_iteration, np.full(n_new_cuts, self.iteration)]
        )
        self.last_active_iteration = np.concatenate(
            [self.last_active_iteration, np.full(n_new_cuts, self.iteration)]
        )
        logger.debug(f"{self.__class__.__name__} added {n_new_cuts} cuts")

    def remove_linear_constraints_hook(self, model, index):
        indexremove.remove(self.linear_constraint_index.ravel(), index.ravel())

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
            self.linear_constraint_index = self.linear_constraint_index[kept]
            self.added_iteration = self.added_iteration[kept]
            self.last_active_iteration = self.last_active_iteration[kept]
            self.cut_coef_unique_list.erase(dropped)
            self.coef = self.coef[kept]
            self.offset = self.offset[kept]
        logger.debug(f"{self.__class__.__name__} removed {dropped.size} cuts")

    def solve_exit_hook(self, model):
        model.assert_optimal(suboptimal=True)
        slacks = model.get_linear_constraint_slacks()[
            self.linear_constraint_index
        ]
        self.last_active_iteration[np.abs(slacks) <= 1e-6] = self.iteration


if __name__ == "__main__":
    import doctest

    n_failuers, _ = doctest.testmod(
        optionflags=doctest.ELLIPSIS + doctest.NORMALIZE_WHITESPACE
    )
    if n_failuers > 0:
        raise ValueError(f"failed {n_failuers} tests")

# vimquickrun: python %
