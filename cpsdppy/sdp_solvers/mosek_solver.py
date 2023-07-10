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
        t = model.variable("x", fusion.Domain.unbounded([c.size]))

        for block_index in range(n_blocks):
            # sum_{j = 1}^m A_{ij} x_j - A_{i0} : PSD
            coef = problem_data["lmi_constraint_coefficient"][block_index]
            offset = problem_data["lmi_constraint_offset"][block_index]
            np.testing.assert_equal(type(coef), list)
            np.testing.assert_equal(len(coef), c.size)
            if c.size > 0:
                np.testing.assert_equal(type(coef[0]), scipy.sparse.csr_matrix)
            np.testing.assert_equal(type(offset), scipy.sparse.csr_matrix)
            matrices = []
            matrices.append(
                fusion.Expr.neg(fusion.Expr.constTerm(offset.todense()))
            )

            for j in range(len(coef)):
                coefj_scipy = coef[j].tocoo()
                coefj_fusion = fusion.Matrix.sparse(
                    coefj_scipy.shape[0],
                    coefj_scipy.shape[1],
                    coefj_scipy.row,
                    coefj_scipy.col,
                    coefj_scipy.data,
                )
                _t = fusion.Expr.reshape(
                    fusion.Expr.repeat(
                        t.index([j]), np.prod(coef[j].shape), 0
                    ),
                    list(coef[j].shape),
                )
                matrices.append(fusion.Expr.mulElm(_t, coefj_fusion))
            matrices = [
                fusion.Expr.reshape(x, np.r_[1, x.getShape()])
                for x in matrices
            ]
            matrices = fusion.Expr.stack(0, matrices)
            model.constraint(
                fusion.Expr.sum(matrices, 0), fusion.Domain.inPSDCone()
            )

        model.objective(fusion.ObjectiveSense.Minimize, fusion.Expr.dot(t, c))

        res = mosek_utils.solve(model, config)

        target_objective = problem_data.get("target_objective", None)
        test = (
            (target_objective is not None)
            and np.isfinite(res["primal_objective"])
            and np.isfinite(res["dual_objective"])
        )
        if test:
            target_tol = 5e-2
            test = True
            try:
                np.testing.assert_allclose(
                    res["primal_objective"],
                    target_objective,
                    rtol=target_tol,
                    atol=target_tol,
                )
            except AssertionError as e:
                print(e)
                test = False
            try:
                np.testing.assert_allclose(
                    res["dual_objective"],
                    target_objective,
                    rtol=target_tol,
                    atol=target_tol,
                )
            except AssertionError as e:
                print(e)
                test = False
            if not test:
                raise ValueError(
                    f"expected {target_objective} but mosek "
                    f"primal objective = {res['primal_objective']} and "
                    f"dual objective = {res['dual_objective']}"
                )

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
