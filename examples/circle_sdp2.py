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
import scipy.sparse

import cpsdppy

# import shutil

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})


logger = logging.getLogger(__name__)

plot_dir = "tmp/circle_sdp/v2"

# if os.path.exists(plot_dir):
#     shutil.rmtree(plot_dir)
os.makedirs(plot_dir, exist_ok=True)

grid_size = 300


def main():
    problem_name = "e"
    cpsdppy.logging_helper.setup()
    config = cpsdppy.config.Config()
    config.initial_cut_type = "none"
    config.eval_lb_every = 0
    config.iteration_limit = 3
    config.step_size = 5.0
    config.eigen_comb_cut = 1
    config.tol = 1e-6
    config.feas_tol = 1e-6
    config.add_cuts_after_optimality_step = 0
    config.eigen_comb_cut = 0

    problem_data = get_problem_data(problem_name)
    problem_data["target_objective"] = cpsdppy.sdp_solvers.mosek_solver.run(
        problem_data, config
    )["primal_objective"]
    problem_data["initial_x"] = -2
    result = cpsdppy.sdp_solvers.subgradient_projection.run(
        problem_data, config
    )

    x_list = np.concatenate(
        [result["initial_x"][None], result["iter_x"]], axis=0
    )
    v_list = result["iter_v"]
    cut_coef = result["cut_coef"]
    cut_offset = result["cut_offset"]
    cut_data = result["cut_data"]
    constr_svec_coefs = problem_data["lmi_svec_constraint_coefficient"][0]
    constr_svec_offset = problem_data["lmi_svec_constraint_offset"][0]

    fig, ax = plt.subplots()
    ax.axis("equal")
    box_color = "none"
    ax.axhline(-2, lw=1, color=box_color)
    ax.axhline(2, lw=1, color=box_color)
    ax.axvline(-2, lw=1, color=box_color)
    ax.axvline(2, lw=1, color=box_color)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    px = np.linspace(*xlim, grid_size)
    py = np.linspace(*ylim, grid_size)

    det = []
    for _y in py:
        for _x in px:
            _xy = np.array([_x, _y])
            Cx = constr_svec_coefs @ _xy - constr_svec_offset
            Cx = cpsdppy.linalg.svec_inv(Cx, part="f")
            det.append(np.linalg.eigh(Cx)[0][0])
    det = np.array(det).reshape(py.size, px.size)
    feasible_region_color = "gray"
    ax.contour(
        px, py, det, levels=[0], colors=[feasible_region_color], linewidths=1
    )

    os.makedirs(plot_dir, exist_ok=True)
    figpath = f"{plot_dir}/sdp_{problem_name}.pdf"
    # default: 6.4, 4.8
    fig.set_size_inches(3.2, 2.4)
    fig.savefig(figpath, dpi=300, transparent=True)
    fig.savefig(figpath.replace("pdf", "png"), dpi=300)
    print(figpath)

    for i, x in enumerate(x_list):
        ax.plot(x[0], x[1], "o", color="C0", markersize=3)

    for i, x in enumerate(v_list):
        ax.plot(x[0], x[1], "o", color="C1", markersize=3)

    def draw_arrow(start, end, margin):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="dimgray",
            lw=0.2,
            zorder=1,
        )
        # _start = (1 - margin) * start + margin * end
        # _end = margin * start + (1 - margin) * end
        # _len = _end - _start
        # ax.arrow(
        #     _start[0],
        #     _start[1],
        #     _len[0],
        #     _len[1],
        #     color="gray",
        #     width=0.0001,
        #     head_width=0.02,
        # )

    for i in range(len(v_list)):
        draw_arrow(start=x_list[i], end=v_list[i], margin=0.05)
        draw_arrow(start=v_list[i], end=x_list[i + 1], margin=0.05)

    label_dict = {1: "min", 2: "comb"}

    for i in range(cut_coef.shape[0]):
        g = cut_coef[i]
        offset = cut_offset[i]

        if np.abs(-offset / g[1]) < np.abs(-offset / g[0]):
            point = (0, offset / g[1])
        else:
            point = (offset / g[0], 0)
        ls = "-" if cut_data[i] == 1 else "--"
        if cut_data[i] in label_dict:
            label = label_dict[cut_data[i]]
            del label_dict[cut_data[i]]
        else:
            label = None
        ax.axline(
            point,
            slope=-g[0] / g[1],
            ls=ls,
            color="gray",
            lw=0.5,
            zorder=0,
            label=label,
        )

    # ax.legend(loc="upper center")

    os.makedirs(plot_dir, exist_ok=True)
    figpath = f"{plot_dir}/sdp_linear_cut_{problem_name}.pdf"
    fig.savefig(figpath, dpi=300, transparent=True)
    fig.savefig(figpath.replace("pdf", "png"), dpi=300)
    print(figpath)


def get_problem_data(problem_name):
    if problem_name == "a":
        objective_coef = np.array([0.0, 1.0])

        a = np.array([[1, 0], [0, -1]], dtype=float)
        b = np.array([[0, 1], [1, 0]], dtype=float)
        c = -np.array([[1, 0], [0, 1]], dtype=float)

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
        c = -np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )

    elif problem_name == "c":
        objective_coef = np.array([0, 1.0])
        a = np.array(
            [
                [0, 1, 0, -1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [-1, 0, 1, 0],
            ],
            dtype=float,
        )
        b = np.array(
            [
                [0, 0, -1, 0],
                [0, 0, 1, -1],
                [-1, 1, 0, 0],
                [0, -1, 0, 0],
            ],
            dtype=float,
        )
        c = -np.array(
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
        c = -np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

    elif problem_name == "e":
        objective_coef = np.array([0.0, 1.0])
        rng = np.random.RandomState(0)
        n = 10
        # a = rng.normal(size=(n, n)) / np.sqrt(n)
        a = rng.normal(size=(n, n)) / n
        for i in range(n):
            for j in range(i, n):
                a[i, j] = a[j, i]
        # b = rng.normal(size=(n, n)) / np.sqrt(n)
        b = rng.normal(size=(n, n)) / n
        for i in range(n):
            for j in range(i, n):
                b[i, j] = b[j, i]
        c = -np.eye(n)

    # constr_coefs = np.stack([a, b])
    constr_coefs = [scipy.sparse.csr_matrix(a), scipy.sparse.csr_matrix(b)]
    constr_offset = scipy.sparse.csr_matrix(c)

    svec_constr_coefs = scipy.sparse.hstack(
        [cpsdppy.linalg.svec(x) for x in constr_coefs], format="csr"
    )
    svec_constr_offset = (
        cpsdppy.linalg.svec(constr_offset).toarray().reshape(-1)
    )

    return {
        "n_variables": 2,
        "objective_sense": "min",
        "objective_offset": 0,
        "variable_lb": np.full(2, -np.inf),
        "variable_ub": np.full(2, np.inf),
        "objective_coefficient": objective_coef,
        "linear_constraint_coefficient": scipy.sparse.csr_matrix(
            ([], ([], [])), shape=(0, 2)
        ),
        "linear_constraint_rhs": np.zeros(0),
        "linear_constraint_sense": np.zeros(0, dtype="U"),
        "lmi_constraint_coefficient": [constr_coefs],
        "lmi_constraint_offset": [constr_offset],
        "lmi_svec_constraint_coefficient": [svec_constr_coefs],
        "lmi_svec_constraint_offset": [svec_constr_offset],
    }


if __name__ == "__main__":
    main()

# vimquickrun: . ./scripts/activate.sh ; python %
