# -*- coding: utf-8 -*-

"""Optimization of a linear function over a circle using SDP

In this example we optimize a linear function over a circle.

min  c^T x
s.t. |x|^2 <= 1.

Denote x = [p q].
The circle is represented as a linear matrix inequality

min  c^T [p q]

     [1+p   q ]
s.t. [        ]  :  PSD
     [ q   1-p]

More specifically, we use the minimum eigenvalue to formulate the constraint

min  c^T [p q]

                 [1+p   q ]
s.t. lambda_min  [        ]  >=  0
                 [ q   1-p]

Let us denote

    [1+p   q ]
S = [        ]
    [ q   1-p]

     [ 1   0 ]     [ 0   1 ]     [ 1   0 ]
   = [       ] p + [       ] q + [       ].
     [ 0  -1 ]     [ 1   0 ]     [ 0   1 ]


   = A p + B q + C.

We note that lambda_min is a concave function.
The subgradient g of lambda_min at bar_x = [bar_p bar_q] is given by

g_p = -v_min(bar_S) v_min(bar_S)^T bullet A
    = -v_min(bar_S) A v_min(bar_S)^T,
g_q = -v_min(bar_S) v_min(bar_S)^T bullet B
    = -v_min(bar_S) B v_min(bar_S)^T,

where v_min(bar_S) is a eigenvector associated with the minimum eigenvalue
lambda_min(bar_S) of bar_S = A bar_p + B bar_q + C.
The liniearised constraint is

lambda_min(bar_S) + g^T (x - bar_x) >= 0

We have

g^T bar_x + v_min(bar_S)^T C v_min(bar_S)
= v_min(bar_S)^T (A bar_p + B bar_q + C) v_min(bar_S)
= lambda_min(bar_S)

Therefore, the linearised constraint can be written as

v_min(bar_S)^T S v_min(bar_S)^T >= 0

Thus, we replaces the positive definiteness constraint with

v^T S v >= 0.

Similarly, we can replace the positive definiteness constraint with

V^T S V >= 0

with V in R^{n x 2}.
This constraint can be written as SOCP.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

import cpsdppy

logger = logging.getLogger(__name__)


def main():
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    sdp_linear_cuts()
    sdp_lmi_cuts()


def sdp_linear_cuts():
    objective_coef = np.array([0.0, 1.0])

    model = cpsdppy.mip_solvers.gurobi.GurobiInterface()

    model.add_variables(lb=-2, ub=2, obj=objective_coef)

    linear_cuts = cpsdppy.mip_solver_extensions.LinearCuts(model)

    a = np.array([[1, 0], [0, -1]], dtype=float)
    b = np.array([[0, 1], [1, 0]], dtype=float)
    c = np.array([[1, 0], [0, 1]], dtype=float)

    constr_coefs = np.stack([a, b])
    constr_svec_coefs = np.stack(
        [cpsdppy.linalg.svec(a), cpsdppy.linalg.svec(b)]
    )
    constr_svec_offset = cpsdppy.linalg.svec(c)

    fig, ax = plt.subplots()
    t = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(t)
    y = np.sin(t)
    ax.plot(x, y, "-", color="lightgray", lw=1)
    ax.axis("equal")
    ax.axhline(-2, lw=1, color="C0")
    ax.axhline(2, lw=1, color="C0")
    ax.axvline(-2, lw=1, color="C0")
    ax.axvline(2, lw=1, color="C0")

    for iteration in range(10):
        linear_cuts.iteration = iteration
        model.solve()
        x = model.get_solution()
        matrix = cpsdppy.linalg.svec_inv(
            x @ constr_svec_coefs + constr_svec_offset, part="f"
        )
        w, v = np.linalg.eigh(matrix)
        f = -w[0]
        v_min = v[:, 0]

        g = np.sum(constr_coefs * v_min, axis=2)
        g = -np.sum(g * v_min, axis=1)

        _offset = f - g @ x
        linear_cuts.add_linear_cuts(coef=g, offset=_offset)

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

        ax.plot(x[0], x[1], "o", color="C0", markersize=4)
        ax.axline(
            (0, -_offset / g[1]),
            slope=-g[0] / g[1],
            color="C0",
            lw=0.5,
            zorder=0,
        )

    os.makedirs("tmp", exist_ok=True)
    fig.savefig("tmp/sdp_linear_cut.pdf", dpi=300)


def sdp_lmi_cuts():
    objective_coef = np.array([0.0, 1.0])

    model = cpsdppy.mip_solvers.gurobi.GurobiInterface()

    model.add_variables(lb=-2, ub=2, obj=objective_coef)

    lmi_cuts = cpsdppy.mip_solver_extensions.LMICuts(model)

    a = np.array([[1, 0], [0, -1]], dtype=float)
    b = np.array([[0, 1], [1, 0]], dtype=float)
    c = np.array([[1, 0], [0, 1]], dtype=float)

    constr_coefs = np.stack([a, b])
    constr_offset = c
    constr_svec_coefs = np.stack(
        [cpsdppy.linalg.svec(a), cpsdppy.linalg.svec(b)]
    )
    constr_svec_offset = cpsdppy.linalg.svec(c)

    fig, ax = plt.subplots()
    t = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(t)
    y = np.sin(t)
    ax.plot(x, y, "-", color="lightgray", lw=1)
    ax.axis("equal")
    ax.axhline(-2, lw=1, color="C0")
    ax.axhline(2, lw=1, color="C0")
    ax.axvline(-2, lw=1, color="C0")
    ax.axvline(2, lw=1, color="C0")
    C_list = []
    offset_list = []

    for iteration in range(3):
        lmi_cuts.iteration = iteration
        model.solve()
        x = model.get_solution()[:2]
        matrix = cpsdppy.linalg.svec_inv(
            x @ constr_svec_coefs + constr_svec_offset, part="f"
        )

        w, v = np.linalg.eigh(matrix)

        coef_v0 = np.sum(constr_coefs * v[:, 0], axis=2)
        coef_v1 = np.sum(constr_coefs * v[:, 1], axis=2)
        coef_v0v0 = -np.sum(coef_v0 * v[:, 0], axis=1)
        coef_v0v1 = -np.sum(coef_v0 * v[:, 1], axis=1)
        coef_v1v1 = -np.sum(coef_v1 * v[:, 1], axis=1)
        cut_coef = -np.stack([coef_v0v0, coef_v0v1, coef_v1v1])
        cut_offset = np.array(
            [
                v[:, 0] @ constr_offset @ v[:, 0],
                v[:, 0] @ constr_offset @ v[:, 1],
                v[:, 1] @ constr_offset @ v[:, 1],
            ]
        )

        obj = objective_coef @ x
        constr = np.sum(x**2) - 1
        n_linear_cuts = 0
        n_lmi_cuts = lmi_cuts.n
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

        ax.plot(x[0], x[1], "o", markersize=4, color="C0")

        lmi_cuts.add_lmi_cuts(coef=cut_coef, offset=cut_offset)
        C_list.append(cut_coef)
        offset_list.append(cut_offset)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    px = np.linspace(*xlim, 50)
    py = np.linspace(*ylim, 50)

    for _C, _offset in zip(C_list, offset_list):
        det = []
        for _x in px:
            for _y in py:
                _xy = np.array([_x, _y])
                Cx = _C @ _xy + _offset
                det.append(Cx[0] * Cx[2] - Cx[1] ** 2)
        det = np.array(det).reshape(px.size, py.size)
        ax.contour(px, py, det, levels=[0], colors=[f"C{0}"], linewidths=1)

    os.makedirs("tmp", exist_ok=True)
    fig.savefig("tmp/sdp_lmi_cut.pdf", dpi=300)


if __name__ == "__main__":
    main()

# vimquickrun: python %
