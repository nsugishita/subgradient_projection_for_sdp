# -*- coding: utf-8 -*-

"""A problem extension with Moreu-Yoshida regularisation"""

import logging
import weakref

import numpy as np

logger = logging.getLogger(__name__)


class MoreuYoshidaRegularisation:
    """A problem extension with Moreu-Yoshida regularisation

    Examples
    --------
    >>> import cpsdppy
    >>> text = '''
    ... Minimize
    ... 8 x1 + 5 x2 + 8 x3
    ... Subject To
    ... 2 x1 + x2 >= 2
    ... x2 + 4 x3 >= 1
    ... Bounds
    ... End
    ... '''
    >>> m = cpsdppy.mip_solvers.gurobi.GurobiInterface.read_string(text)
    >>> m.solve()
    >>> m.get_solution()
    array([0.5, 1. , 0. ])

    >>> m = cpsdppy.mip_solvers.gurobi.GurobiInterface.read_string(text)
    >>> reg = MoreuYoshidaRegularisation(m)
    >>> x = np.array([0.5, 0.5, 0.5])
    >>> for i in range(8):
    ...     x = reg.prox(x)
    ...     print(f'{x[0]:8.4f}  {x[1]:8.4f}  {x[2]:8.4f}')
    0.7000    0.6000    0.1000
    0.6619    0.6762    0.0810
    0.6238    0.7524    0.0619
    0.5857    0.8286    0.0429
    0.5476    0.9048    0.0238
    0.5095    0.9810    0.0048
    0.5000    1.0000    0.0000
    0.5000    1.0000    0.0000

    >>> reg.project([0, 1, 0]).round(2)
    array([0.4, 1.2, 0. ])
    """

    def __init__(self, model, config=None):
        """Initialise a MoreuYoshidaRegularisation instance"""
        # self.model = weakref.ref(model)
        self.obj_coefs = model.get_linear_objective_coefs()
        self.config = config
        self.model = weakref.ref(model)
        model.add_hooks(self)
        self.step_size = 0.1
        self.proximal_centre = np.zeros(model.get_n_variables())

    def project(self, x=None):
        """Project the given point to the feasible region

        This finds the projection of the given point to the feasible region.
        """
        if x is not None:
            self.set_proximal_centre(x)
        model = self.model()
        original_coefs = self.obj_coefs
        self.obj_coefs = np.zeros_like(self.obj_coefs)
        model.solve()
        x = model.get_solution()[: len(x)]
        self.obj_coefs = original_coefs
        return x

    def prox(self, x=None):
        """Solve with proximal regularisation centred at the given point"""
        if x is not None:
            self.set_proximal_centre(x)
        model = self.model()
        model.solve()
        x = model.get_solution()[: self.proximal_centre.size]
        return x

    def set_proximal_centre(self, x):
        self.proximal_centre = np.asarray(x)

    def solve_enter_hook(self, model):
        logger.debug(f"{self.__class__.__name__} enterhook")
        self.original_step_size = self.step_size

    def solve_prehook(self, model):
        logger.debug(f"{self.__class__.__name__} prehook")
        ss = self.step_size
        centre = self.proximal_centre
        c = self.obj_coefs
        ss_inv = 1 / (2 * ss)
        vars = np.array(model.model.getVars())[: centre.size]
        model.model.setObjective(
            c @ vars + ss_inv * (vars - centre) @ (vars - centre)
        )

    def solve_posthook(self, model):
        logger.debug(f"{self.__class__.__name__} posthook")
        if model.is_optimal(suboptimal=True):
            return
        self.step_size /= 2
        return True

    def solve_exit_hook(self, model):
        logger.debug(f"{self.__class__.__name__} exithook")
        self.step_size = self.original_step_size


if __name__ == "__main__":
    import doctest

    n_failuers, _ = doctest.testmod(
        optionflags=doctest.ELLIPSIS + doctest.NORMALIZE_WHITESPACE
    )
    if n_failuers > 0:
        raise ValueError(f"failed {n_failuers} tests")

# vimquickrun: python %
