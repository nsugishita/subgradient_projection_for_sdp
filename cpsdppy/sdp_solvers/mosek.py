# -*- coding: utf-8 -*-

"""Solve SDP using Mosek"""


def run(problem_data, config):
    import mosek.fusion as fusion
    import numpy as np
    import scipy.sparse

    from cpsdppy.utils import mosek_utils

    n_blocks = len(problem_data["lmi_constraint_coefficient"])
    c = np.asarray(problem_data["objective_coefficient"])

    with fusion.Model() as model:
        block_sizes = [
            problem_data["lmi_constraint_offset"][block_index].shape[0]
            for block_index in range(n_blocks)
        ]

        X = [
            model.variable(
                "X", fusion.Domain.inPSDCone(block_sizes[block_index])
            )
            for block_index in range(n_blocks)
        ]

        constraints = []

        for i in range(len(c)):
            lhs = 0
            for block_index in range(n_blocks):
                F = problem_data["lmi_constraint_coefficient"][block_index][
                    i
                ].tocoo()
                coef = fusion.Matrix.sparse(
                    F.shape[0], F.shape[1], F.row, F.col, F.data
                )
                lhs = fusion.Expr.add(
                    lhs, fusion.Expr.dot(coef, X[block_index])
                )
            constraints.append(
                model.constraint(lhs, fusion.Domain.equalsTo(c[i]))
            )

        obj = 0
        for block_index in range(n_blocks):
            F = problem_data["lmi_constraint_offset"][block_index].tocoo()
            coef = fusion.Matrix.sparse(
                F.shape[0], F.shape[1], F.row, F.col, F.data
            )
            obj = fusion.Expr.add(obj, fusion.Expr.dot(coef, X[block_index]))
        model.objective(fusion.ObjectiveSense.Maximize, obj)

        res = mosek_utils.solve(model, config)
        # res["x"] = [x_.level() for x_ in X]
        res["y"] = np.array([c_.dual() for c_ in constraints]).reshape(-1)

        # target_objective = problem_data.get("target_objective", None)
        # test = (
        #     (target_objective is not None)
        #     and np.isfinite(res["primal_objective"])
        #     and np.isfinite(res["dual_objective"])
        # )
        # if test:
        #     target_tol = 5e-2
        #     test = True
        #     try:
        #         np.testing.assert_allclose(
        #             res["primal_objective"],
        #             target_objective,
        #             rtol=target_tol,
        #             atol=target_tol,
        #         )
        #     except AssertionError as e:
        #         print(e)
        #         test = False
        #     try:
        #         np.testing.assert_allclose(
        #             res["dual_objective"],
        #             target_objective,
        #             rtol=target_tol,
        #             atol=target_tol,
        #         )
        #     except AssertionError as e:
        #         print(e)
        #         test = False
        #     if not test:
        #         raise ValueError(
        #             f"expected {target_objective} but mosek "
        #             f"primal objective = {res['primal_objective']} and "
        #             f"dual objective = {res['dual_objective']}"
        #         )

        return res


def main():
    from cpsdppy import config as config_module
    from cpsdppy import sdpa

    config = config_module.Config()
    config.problem_name = "theta1"
    config.log_to_stdout = 1
    # config.tol = 1e-6
    # config.feas_tol = 1e-6
    config._display_non_default()
    problem_data = sdpa.read(config)
    res = run(problem_data, config)
    print(f"obj: {res['primal_objective']:.2f}")
    print(f"time: {res['walltime']:.2f}")


if __name__ == "__main__":
    main()

# vimquickrun: . ./scripts/activate.sh ; python %
