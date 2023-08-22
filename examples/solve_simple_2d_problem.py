# -*- coding: utf-8 -*-

"""Run the subgradient projection method on a simple problem"""

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse

import cpsdppy

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

plot_dir = "outputs/simple_example/v1"

os.makedirs(plot_dir, exist_ok=True)

grid_size = 300


def main():
    config = cpsdppy.config.Config()

    config.eval_lb_every = 0
    config.iteration_limit = 3
    config.step_size = 2.0
    config.tol = 1e-6
    config.feas_tol = 1e-6
    config.add_cuts_after_optimality_step = 0
    config.n_linear_cuts = 1
    config.eigen_comb_cut = 0
    initial_x = [-3, -3]

    problem_data = get_problem_data()
    problem_data["target_objective"] = cpsdppy.sdp_solvers.mosek_solver.run(
        problem_data, config
    )["primal_objective"]
    problem_data["initial_x"] = np.array(initial_x)
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
    ax.axhline(-5, lw=1, color=box_color)
    ax.axhline(4, lw=1, color=box_color)
    ax.axvline(-4, lw=1, color=box_color)
    ax.axvline(4, lw=1, color=box_color)

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
    figpath = f"{plot_dir}/feasible_set.pdf"
    # default: 6.4, 4.8
    fig.set_size_inches(3.2, 2.4)
    fig.savefig(figpath, dpi=300, transparent=True)
    fig.savefig(figpath.replace("pdf", "png"), dpi=300)
    print(figpath)

    for i, x in enumerate(x_list):
        ax.scatter(
            x[0],
            x[1],
            s=3,
            marker="o",
            edgecolors="gray",
            c="gray",
            linewidths=0.4,
            zorder=99,
        )

    for i, x in enumerate(v_list):
        ax.scatter(
            x[0],
            x[1],
            s=3,
            marker="o",
            edgecolors="gray",
            c="white",
            linewidths=0.4,
            zorder=99,
        )

    def draw_arrow(start, end, margin):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="dimgray",
            ls=(0, (5, 5)),
            lw=0.2,
            zorder=1,
        )

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

    x_offsets = [
        [-0.2, 0.4],
        [-0.7, 0.6],
        [1.1, -0.1],
        [1.1, 0.0],
    ]
    x_box = [False, False, False, False]
    v_offsets = [
        [-0.5, -0.8],
        [0.7, -0.5],
        [1.1, 0.2],
    ]
    v_box = [False, False, True]
    bbox = dict(boxstyle="square,pad=0.00", ec="none", fc="white", alpha=0.8)

    for i in range(len(x_list)):
        ax.text(
            x_list[i][0] + x_offsets[i][0],
            x_list[i][1] + x_offsets[i][1],
            "$x^{(" + f"{i}" + ")}$",
            va="center",
            ha="center",
            bbox=bbox if x_box[i] else None,
        )
    for i in range(len(v_list)):
        ax.text(
            v_list[i][0] + v_offsets[i][0],
            v_list[i][1] + v_offsets[i][1],
            "$y^{(" + f"{i}" + ")}$",
            va="center",
            ha="center",
            bbox=bbox if v_box[i] else None,
        )

    ax.set_xticks([])
    ax.set_yticks([])

    os.makedirs(plot_dir, exist_ok=True)
    figpath = f"{plot_dir}/iterates.pdf"
    fig.savefig(figpath, dpi=300, transparent=True)
    fig.savefig(figpath.replace("pdf", "png"), dpi=300)
    print(figpath)


def get_problem_data():
    objective_coef = np.array([0.0, 1.0])
    rng = np.random.RandomState(1)
    n = 3
    a = rng.normal(size=(n, n)) / n
    for i in range(n):
        for j in range(i, n):
            a[i, j] = a[j, i]
    b = rng.normal(size=(n, n)) / n
    for i in range(n):
        for j in range(i, n):
            b[i, j] = b[j, i]
    c = -np.eye(n)

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
