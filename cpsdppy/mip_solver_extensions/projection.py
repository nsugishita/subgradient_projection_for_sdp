# -*- coding: utf-8 -*-

"""A problem extension with Moreu-Yoshida regularisation"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def project(model, x):
    """Project the given point to the feasible region

    This finds the projection of the given point to the feasible region.

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
    >>> m = cpsdppy.mip_solvers.read(text)
    >>> m.solve()
    >>> m.get_solution().round(2)
    array([0.5, 1. , 0. ])
    >>> project(m, [0, 1, 0]).round(2)
    array([0.4, 1.2, 0. ])
    >>> m.solve()
    >>> m.get_solution().round(2)
    array([0.5, 1. , 0. ])
    """
    x = np.asarray(x)
    original_linear_objective_coefs = model.get_linear_objective_coefs()
    regularisation = 1.0
    for i in range(10):
        _set_objective(
            model=model,
            type="quadratic",
            lin_cost=0.0,
            quad_cost=regularisation,
            centre=x,
        )
        model.solve()
        if model.is_optimal(suboptimal=False):
            break
        # Failed to solve the model, maybe due to numerical difficulties.
        # Tighten the regularisation and try again.
        regularisation *= 2
    projection = model.get_solution()[: len(x)]
    # TODO Use hooks to properly process variable deletion.
    original_linear_objective_coefs = original_linear_objective_coefs[
        : model.get_n_variables()
    ]
    _set_objective(
        model=model,
        type="linear",
        lin_cost=original_linear_objective_coefs,
        quad_cost=None,
        centre=None,
    )
    return projection


def _set_objective(model, type, lin_cost, quad_cost, centre):
    """Update objective to `lin_cost x + quad_cost / 2 (x - centre)^2`"""
    assert type in ["linear", "quadratic"]
    if centre is None:
        centre = np.arange(model.get_n_variables())
    lin_cost = np.broadcast_to(lin_cost, centre.shape)
    solver_name = model.__class__.__name__.lower()
    # sign = 1 if self.problem_data["objective_sense"] == "min" else -1
    if "gurobi" in solver_name:
        vars = np.array(model.model.getVars())[: centre.size]
        if type == "quadratic":
            model.model.setObjective(
                lin_cost @ vars
                + quad_cost / 2 * (vars - centre) @ (vars - centre)
            )
        else:
            model.model.setObjective(lin_cost @ vars)
    elif solver_name == "cplex":
        vars = np.arange(centre.size)
        if type == "quadratic":
            model.set_objective_quadratic_coefficient(vars, vars, quad_cost)
            model.set_objective_coefficient(
                vars, lin_cost - centre * quad_cost
            )
        else:
            model.set_objective_coefficient(vars, lin_cost)
    elif "cplex" in solver_name:
        if type == "quadratic":
            # This function makes the problem type to QP even if
            # the coefficient is 0. Maybe we want to avoid calling it if not
            # necessary?

            # Becareful with the behavior of set_quadratic_coefficients!
            # If one calls set_quadratic_coefficients(i, j, v) with distinct
            # i, j, then the objective will be
            #    obj = [v x_i x_j + v x_j x_i] / 2,
            # or simply
            #    obj = [2.0 v x_i x_j] / 2 = v x_i x_j.
            # If i and j is equal, then
            #    obj = [v x_i x_i] / 2.
            # So, if i and j is different, the coefficient will be untouched,
            # BUT IF I AND J IS EQUAL, THE COEFFICIENT WILL BE DIVIDED BY 2.
            vars = np.arange(centre.size)
            iter = enumerate(map(int, vars))
            model.model.objective.set_quadratic_coefficients(
                [(xi, xi, quad_cost) for i, xi in iter]
            )
            iter = enumerate(
                zip(
                    map(int, vars),
                    lin_cost,
                    centre,
                )
            )
            model.model.objective.set_linear(
                [
                    (
                        xi,
                        float(_lin_cost - _centre * quad_cost),
                    )
                    for i, (xi, _lin_cost, _centre) in iter
                ]
            )
        else:
            model.model.objective.set_linear(zip(map(int, vars), lin_cost))
    else:
        raise ValueError(f"unknown solver name: {solver_name}")


if __name__ == "__main__":
    import doctest

    n_failuers, _ = doctest.testmod(
        optionflags=doctest.ELLIPSIS + doctest.NORMALIZE_WHITESPACE
    )
    if n_failuers > 0:
        raise ValueError(f"failed {n_failuers} tests")

# vimquickrun: python %
