# -*- coding: utf-8 -*-

"""Optimization of a linear function over a circle

In this example we optimize a linear function over a circle.
The circle is defined as a level set of a quadratic function:

min  c^T x
s.t. |x|^2 <= 1.
"""

import logging

import numpy as np

import cpsdppy

logger = logging.getLogger(__name__)


def main():
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    logger.info("unregularised")
    circle_linear_cuts()
    logger.info("regularised")
    circle_linear_cuts_regularised()


def circle_linear_cuts():
    objective_coef = np.array([-1.0, 0.0])

    model = cpsdppy.mip_solvers.gurobi.GurobiInterface()

    model.add_variables(shape=2, lb=-2, ub=2, obj=[-1.0, -0.0])

    linear_cuts = cpsdppy.mip_solver_extensions.LinearCuts(model)

    for iteration in range(20):
        linear_cuts.iteration = iteration

        model.solve()
        x = model.get_solution()
        f = np.sum(x**2) - 1
        g = 2 * x
        offset = f - g @ x
        linear_cuts.add_linear_cuts(coef=g, offset=offset)

        obj = objective_coef @ x
        constr = np.sum(x**2) - 1
        n_linear_cuts = linear_cuts.n
        n_lmi_cuts = 0
        if iteration == 0:
            logger.info(
                f"{'it':>3s} "
                f"{'obj':>6s} {'constr':>6s} {'x0':>7s} {'x1':>7s} "
                f"{'lnrcuts'} {'lmicuts'}"
            )
        logger.info(
            f"{iteration + 1:3d} "
            f"{obj:6.2f} {constr:6.2f} {x[0]:7.4f} {x[1]:7.4f} "
            f"{n_linear_cuts:7d} {n_lmi_cuts:7d}"
        )

    np.testing.assert_allclose(
        x,
        -objective_coef / np.linalg.norm(objective_coef, 2),
        rtol=1e-2,
        atol=1e-2,
    )


def circle_linear_cuts_regularised():
    objective_coef = np.array([-1.0, 0.0])

    model = cpsdppy.mip_solvers.gurobi.GurobiInterface()

    model.add_variables(shape=2, lb=-2, ub=2, obj=[-1.0, -0.0])

    reg = cpsdppy.mip_solver_extensions.MoreuYoshidaRegularisation(
        model, config=None
    )
    reg.step_size = 0.4
    linear_cuts = cpsdppy.mip_solver_extensions.LinearCuts(model)

    x = np.array([2, -2])

    for iteration in range(20):
        linear_cuts.iteration = iteration

        f = np.sum(x**2) - 1
        g = 2 * x
        offset = f - g @ x
        linear_cuts.add_linear_cuts(coef=g, offset=offset)

        if f > 0:
            relaxation_parameter = 1.0
            x = x - relaxation_parameter * f * g / np.linalg.norm(g) ** 2
        x = reg.project(x)

        x = reg.prox(x)

        obj = objective_coef @ x
        constr = np.sum(x**2) - 1
        n_linear_cuts = linear_cuts.n
        n_lmi_cuts = 0
        if iteration == 0:
            logger.info(
                f"{'it':>3s} "
                f"{'obj':>6s} {'constr':>6s} {'x0':>7s} {'x1':>7s} "
                f"{'lnrcuts'} {'lmicuts'}"
            )
        logger.info(
            f"{iteration + 1:3d} "
            f"{obj:6.2f} {constr:6.2f} {x[0]:7.4f} {x[1]:7.4f} "
            f"{n_linear_cuts:7d} {n_lmi_cuts:7d}"
        )

    np.testing.assert_allclose(
        x,
        -objective_coef / np.linalg.norm(objective_coef, 2),
        rtol=1e-2,
        atol=1e-2,
    )


#
if __name__ == "__main__":
    main()

# vimquickrun: python %
