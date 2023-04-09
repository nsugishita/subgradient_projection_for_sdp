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

import matplotlib as mpl
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
        [cpsdppy.linalg.svec(a), cpsdppy.linalg.svec(b)], axis=1
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
            constr_svec_coefs @ x + constr_svec_offset, part="f"
        )
        w, v = np.linalg.eigh(matrix)
        f = -w[0]
        v_min = v[:, 0]

        g = np.sum(constr_coefs * v_min, axis=2)
        g = -np.sum(g * v_min, axis=1)

        _offset = f - g @ x
        linear_cuts.add_linear_cuts(coef=g, offset=_offset)

        obj = objective_coef @ x
        constr = -w[0]
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
    problem_names = "abcd"
    problem_names = "c"
    for problem_name in problem_names:
        if problem_name == "a":
            objective_coef = np.array([0.0, 1.0])

            a = np.array([[1, 0], [0, -1]], dtype=float)
            b = np.array([[0, 1], [1, 0]], dtype=float)
            c = np.array([[1, 0], [0, 1]], dtype=float)

        elif problem_name == "b":
            objective_coef = np.array([0.0, 1.0])

            a = np.array(
                [
                    [0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0],
                ],
                dtype=float,
            )
            b = np.array(
                [
                    [0, 0, 0.8],
                    [0, 0, 0],
                    [0.8, 0, 0],
                ],
                dtype=float,
            )
            b = np.array(
                [
                    [0, 0, 1],
                    [0, 0, 0],
                    [1, 0, 0],
                ],
                dtype=float,
            )
            c = np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                dtype=float,
            )

        elif problem_name == "c":
            objective_coef = np.array([-1, 0])
            a = np.array(
                [
                    [0, 0, 1, 0],
                    [0, 0, -1, 1],
                    [1, -1, 0, 0],
                    [0, 1, 0, 0],
                ],
                dtype=float,
            )
            b = np.array(
                [
                    [0, 1, 0, -1],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [-1, 0, 1, 0],
                ],
                dtype=float,
            )
            c = np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=float,
            )

        elif problem_name == "d":
            objective_coef = np.array([0.0, 1.0])
            a = np.array(
                [
                    [0, 0, 1, 0],
                    [0, 0, -0.5, 1],
                    [1, -0.5, 0, 0],
                    [0, 1, 0, 0],
                ],
                dtype=float,
            )
            b = np.array(
                [
                    [0, 0.5, 0, -1],
                    [0.5, 0, 0.5, 0],
                    [0, 0.5, 0, 0.5],
                    [-1, 0, 0.5, 0],
                ],
                dtype=float,
            )
            c = np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=float,
            )

        constr_coefs = np.stack([a, b])
        constr_offset = c
        _sdp_lmi_cuts_impl(
            problem_name, objective_coef, constr_coefs, constr_offset
        )


def _sdp_lmi_cuts_impl(
    problem_name, objective_coef, constr_coefs, constr_offset
):
    result = _sdp_lmi_cuts_solver(objective_coef, constr_coefs, constr_offset)
    x_list = result["x_list"]
    lmi_cuts = result["lmi_cuts"]
    constr_svec_coefs = result["constr_svec_coefs"]
    constr_svec_offset = result["constr_svec_offset"]

    fig, ax = plt.subplots()
    box_color = "lightgray"
    ax.axhline(-2, lw=1, color=box_color)
    ax.axhline(2, lw=1, color=box_color)
    ax.axvline(-2, lw=1, color=box_color)
    ax.axvline(2, lw=1, color=box_color)

    fig_it_by_it, ax_it_by_it = plt.subplots(2, 2)
    for _ax in ax_it_by_it.ravel():
        _ax.axhline(-2, lw=1, color=box_color)
        _ax.axhline(2, lw=1, color=box_color)
        _ax.axvline(-2, lw=1, color=box_color)
        _ax.axvline(2, lw=1, color=box_color)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    px = np.linspace(*xlim, 300)
    py = np.linspace(*ylim, 300)

    for iteration, x in enumerate(x_list):
        iterate_color = "C0"
        ax.plot(x[0], x[1], "o", markersize=4, color=iterate_color)
        _ax = ax_it_by_it.ravel()[iteration]
        _ax.plot(x[0], x[1], "o", markersize=4, color=iterate_color)

    for i in range(lmi_cuts.n):
        _C = lmi_cuts.coef[3 * i : 3 * i + 3]
        _offset = lmi_cuts.offset[i]
        det = []
        for _y in py:
            for _x in px:
                _xy = np.array([_x, _y])
                Cx = _C @ _xy + _offset
                buf = np.min([Cx[0] * Cx[2] - Cx[1] ** 2, Cx[0], Cx[2]])
                det.append(buf)
        det = np.array(det).reshape(px.size, py.size)

        color = mpl.colors.to_rgb(f"C{i + 1}")
        if i < ax_it_by_it.size - 1:
            plot_lmi_cut_line(ax, px, py, det, color=color)
            for _ax in ax_it_by_it.ravel()[i + 1 :]:
                plot_lmi_cut_line(_ax, px, py, det, color=color)

    det = []
    for _y in py:
        for _x in px:
            _xy = np.array([_x, _y])
            Cx = constr_svec_coefs @ _xy + constr_svec_offset
            Cx = cpsdppy.linalg.svec_inv(Cx, part="f")
            det.append(np.linalg.eigh(Cx)[0][0])
    det = np.array(det).reshape(py.size, px.size)
    feasible_region_color = "gray"
    ax.contour(
        px, py, det, levels=[0], colors=[feasible_region_color], linewidths=1
    )
    for i, _ax in enumerate(ax_it_by_it.ravel()):
        _ax.contour(
            px,
            py,
            det,
            levels=[0],
            colors=[feasible_region_color],
            linewidths=1,
        )
        _ax.set_xlim(*xlim)
        _ax.set_ylim(*ylim)
        _ax.set_title(f"iteration {i + 1}")
        _ax.axis("equal")
    for _ax in ax_it_by_it[0, :]:
        _ax.set_xticklabels([])
    ax.axis("equal")

    os.makedirs("tmp", exist_ok=True)
    fig.savefig(f"tmp/sdp_lmi_cut_{problem_name}.pdf", dpi=300)
    fig_it_by_it.savefig(
        f"tmp/sdp_lmi_cut_it_by_it_{problem_name}.pdf", dpi=300
    )


def plot_lmi_cut_line(ax, px, py, matrix, color):
    ax.contour(px, py, matrix, levels=[0], colors=[color], linewidths=1)


def plot_lmi_cut_area(ax, px, py, matrix, color):
    zeroonematrix = np.full(matrix.shape, np.nan)
    zeroonematrix[matrix < 0] = 1
    zeroonematrix[matrix >= 0] = 0
    cmap = mpl.colors.ListedColormap([(0, 0, 0, 0), color])
    ax.imshow(
        zeroonematrix,
        interpolation="nearest",
        aspect="auto",
        cmap=cmap,
        extent=[px[0], px[-1], py[0], py[-1]],
        origin="lower",
    )


def _sdp_lmi_cuts_solver(objective_coef, constr_coefs, constr_offset):
    constr_svec_coefs = np.stack(
        [cpsdppy.linalg.svec(x) for x in constr_coefs], axis=1
    )
    constr_svec_offset = cpsdppy.linalg.svec(constr_offset)

    model = cpsdppy.mip_solvers.gurobi.GurobiInterface()

    model.add_variables(lb=-2, ub=2, obj=objective_coef)

    lmi_cuts = cpsdppy.mip_solver_extensions.LMICuts(model)

    n_iterations = 4
    x_list = []

    for iteration in range(n_iterations):
        lmi_cuts.iteration = iteration
        model.solve()
        x = model.get_solution()[:2]
        matrix = cpsdppy.linalg.svec_inv(
            constr_svec_coefs @ x + constr_svec_offset, part="f"
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
        constr = -w[0]
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
        x_list.append(x)

        lmi_cuts.add_lmi_cuts(coef=cut_coef, offset=cut_offset)

    return {
        "x_list": x_list,
        "lmi_cuts": lmi_cuts,
        "constr_svec_coefs": constr_svec_coefs,
        "constr_svec_offset": constr_svec_offset,
    }


if __name__ == "__main__":
    main()

# vimquickrun: python %
