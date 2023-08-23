# -*- coding: utf-8 -*-

"""Base class of MILP Solvers"""

import numpy as np


class BaseSolverInterface:
    """Base class of MILP Solvers

    This is the base class of all the MILP solvers.
    This provides hooks which are called on various events, such
    as before/after solving the problem. Below is the list of hooks supported.

    - remove_variables_hooks
        These are called when variables are deleted.
    - remove_linear_constraints_hooks
        These are called when linear constraints are deleted.
    - remove_quadratic_constraints_hooks
        These are called when quadratic constraints are deleted.
    - solve_enter_hooks
        These are called before solving the problem.
    - solve_prehooks
        These are called before solving the problem (after
        `solve_enter_hooks`). When the problem is re-solved (because one of
        `solve_prehooks` returns True), these hooks are called again.
    - solve_posthooks
        These are called after solving the problem (after `solve_enter_hooks`).
        When one of these hooks returns True, the problem is re-solved.
    - solve_exit_hooks
        These are called after solving the problem (after `solve_posthooks`).
        When the problem is re-solved (because one of `solve_posthooks`
        returns True), these are not called.

    Use `add_hooks` to register a new hook.
    """

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
        """Add hooks

        This looks for attributes of the following names on a given object
        and register the found attributes as hooks.

        - remove_variables_hook
        - remove_linear_constraints_hook
        - remove_quadratic_constraints_hook
        - solve_enter_hook
        - solve_prehook
        - solve_posthook
        - solve_exit_hook

        Parameters
        ----------
        hooks : obj
        """
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
        """Remove variables and call hooks"""
        index = np.asarray(index)
        if index.dtype == bool:
            np.testing.assert_equal(index.ndim, 1)
            index = np.nonzero(index)[0]
        self._remove_variables_impl(index)
        for hook in self.remove_variables_hooks:
            hook(self, index)

    def remove_linear_constraints(self, index):
        """Remove linear constraints and call hooks"""
        index = np.asarray(index)
        if index.dtype == bool:
            np.testing.assert_equal(index.ndim, 1)
            index = np.nonzero(index)[0]
        self._remove_linear_constraints_impl(index)
        for hook in self.remove_linear_constraints_hooks:
            hook(self, index)

    def remove_quadratic_constraints(self, index):
        """Remove quadratic constraints and call hooks"""
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
