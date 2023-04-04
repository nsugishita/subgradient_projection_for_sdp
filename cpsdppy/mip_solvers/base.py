# -*- coding: utf-8 -*-

"""Base class of LP Solvers"""

import numpy as np


class BaseSolverInterface:
    """Base class of LP Solvers"""

    def __init__(self, config=None):
        """Initialise a BaseSolverInterface instance"""
        self.solve_enter_hooks = []
        self.solve_prehooks = []
        self.solve_posthooks = []
        self.solve_exit_hooks = []
        self.remove_variables_hooks = []
        self.remove_linear_constraints_hooks = []
        self.remove_quadratic_constraints_hooks = []
        self.variable_index_lists = []
        self.linear_constraint_index_lists = []
        self.quadratic_constraint_index_lists = []

    def add_hooks(self, hooks):
        try:
            self.remove_variables_hooks.append(hooks.remove_variables_hook)
        except AttributeError:
            pass
        try:
            self.remove_linear_constraints_hooks.append(
                hooks.remove_linear_constraints_hook
            )
        except AttributeError:
            pass
        try:
            self.remove_quadratic_constraints_hooks.append(
                hooks.remove_quadratic_constraints_hook
            )
        except AttributeError:
            pass
        try:
            self.solve_enter_hooks.append(hooks.solve_enter_hook)
        except AttributeError:
            pass
        try:
            self.solve_prehooks.append(hooks.solve_prehook)
        except AttributeError:
            pass
        try:
            self.solve_posthooks.append(hooks.solve_posthook)
        except AttributeError:
            pass
        try:
            self.solve_exit_hooks.append(hooks.solve_exit_hook)
        except AttributeError:
            pass

    def remove_variables(self, index):
        index = np.asarray(index)
        if index.dtype == bool:
            np.testing.assert_equal(index.ndim, 1)
            index = np.nonzero(index)[0]
        self._remove_variables_impl(index)
        for hook in self.remove_variables_hooks:
            hook(self, index)

    def remove_linear_constraints(self, index):
        index = np.asarray(index)
        if index.dtype == bool:
            np.testing.assert_equal(index.ndim, 1)
            index = np.nonzero(index)[0]
        self._remove_linear_constraints_impl(index)
        for hook in self.remove_linear_constraints_hooks:
            hook(self, index)

    def remove_quadratic_constraints(self, index):
        index = np.asarray(index)
        if index.dtype == bool:
            np.testing.assert_equal(index.ndim, 1)
            index = np.nonzero(index)[0]
        self._remove_quadratic_constraints_impl(index)
        for hook in self.remove_quadratic_constraints_hooks:
            hook(self, index)

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
