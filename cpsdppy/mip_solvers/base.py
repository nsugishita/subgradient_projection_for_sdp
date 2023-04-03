# -*- coding: utf-8 -*-

"""Base class of LP Solvers"""

import indexremove
import numpy as np


class BaseSolverInterface:
    """Base class of LP Solvers"""

    def __init__(self, config=None):
        """Initialise a BaseSolverInterface instance"""
        self.solve_enter_hooks = []
        self.solve_prehooks = []
        self.solve_posthooks = []
        self.solve_exit_hooks = []
        self.variable_index_lists = []
        self.linear_constraint_index_lists = []
        self.quadratic_constraint_index_lists = []

    def remove_variables(self, index):
        index = np.asarray(index)
        if index.dtype == bool:
            np.testing.assert_equal(index.ndim, 1)
            index = np.nonzero(index)[0]
        self._remove_variables_impl(index)
        for lst in self.variable_index_lists:
            indexremove.remove(lst, index)

    def remove_linear_constraints(self, index):
        index = np.asarray(index)
        if index.dtype == bool:
            np.testing.assert_equal(index.ndim, 1)
            index = np.nonzero(index)[0]
        self._remove_linear_constraints_impl(index)
        for lst in self.linear_constraint_index_lists:
            indexremove.remove(lst, index)

    def remove_quadratic_constraints(self, index):
        index = np.asarray(index)
        if index.dtype == bool:
            np.testing.assert_equal(index.ndim, 1)
            index = np.nonzero(index)[0]
        self._remove_quadratic_constraints_impl(index)
        for lst in self.quadratic_constraint_index_lists:
            indexremove.remove(lst, index)

    def solve(self):
        """Solve the problem and call hooks"""
        i = 0
        posthook_patience = 10
        for hook in self.solve_enter_hooks:
            hook(self)
        while True:
            for hook in self.solve_prehooks:
                hook(self)
            self._solve_impl()
            resolve = False
            for hook in self.solve_posthooks:
                ret = hook(self)
                if ret is True:
                    resolve = True
            if not resolve:
                break
            else:
                i += 1
            if i >= posthook_patience:
                raise ValueError(f"posthook failed for {i} times")
        for hook in self.solve_exit_hooks:
            hook(self)

    def _solve_impl(self):
        raise NotImplementedError(
            "subclass of BaseSolverInterface must implement _solve_impl"
        )
