# -*- coding: utf-8 -*-

"""Lazy constraints"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class LazyConstraints:
    r"""Lazy constraints

    >>> import cpsdppy
    >>> m = cpsdppy.mip_solvers.gurobi_interface.GurobiInterface()
    >>> _ = m.add_variables(shape=2, obj=[1, 2])
    >>> text = '''
    ... 2 x1 + 2 x2 >= 1
    ... 8 x1 + x2 >= 8
    ... 2 x1 + x2 >= 3
    ... x1 + 4 x2 >= 4
    ... x1 + 10 x2 >= 5
    ... '''
    >>> _ = LazyConstraints.read_string(m, config={}, text=text)

    >>> m.print()
    \ LP format - ...
    Minimize
      C0 + 2 C1
    Subject To
    Bounds
    End

    >>> m.solve()

    >>> m.print()
    \ LP format - ...
    Minimize
      C0 + 2 C1
    Subject To
     R0: 8 C0 + C1 >= 8
     R1: C0 + 10 C1 >= 5
     R2: C0 + 4 C1 >= 4
     R3: 2 C0 + C1 >= 3
    Bounds
    End
    """

    def __init__(self, model, coefs, b, config=None):
        """Initialise a LazyConstraints instance"""
        self.coefs = coefs
        self.b = b
        model.add_hooks(self)

    @classmethod
    def read_string(cls, model, config, text):
        header = """
        Minimize
        Subject To
        """
        footer = """
        Bounds
        End
        """
        m = model.read_string(header + text + footer)
        coefs = m.get_linear_constraint_coefs()
        rhs = m.get_linear_constraint_rhs()
        return LazyConstraints(model=model, config=config, coefs=coefs, b=rhs)

    def solve_posthook(self, model):
        if not model.is_optimal():
            raise ValueError(
                "lazy constraing cannot handle an infeasible/unbounded "
                f"{model.get_status_name()}"
            )
        x = model.get_solution()
        slacks = self.coefs @ x - self.b

        if np.all(slacks > 0):
            return

        j = np.argmin(slacks)

        violated = np.array([j])
        satisfied = np.setdiff1d(np.arange(slacks.size), violated)
        model.add_linear_constraints(
            coef=self.coefs[violated], rhs=self.b[violated], sense="G"
        )
        self.coefs = self.coefs[satisfied]
        self.b = self.b[satisfied]

        return True


if __name__ == "__main__":
    import doctest

    n_failuers, _ = doctest.testmod(
        optionflags=doctest.ELLIPSIS + doctest.NORMALIZE_WHITESPACE
    )
    if n_failuers > 0:
        raise ValueError(f"failed {n_failuers} tests")

# vimquickrun: python %
