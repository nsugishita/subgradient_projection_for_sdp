# -*- coding: utf-8 -*-

"""Description of this file"""

import numpy as np
import scipy.sparse

import cpsdppy


def test_svec_numpy():
    x = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 2.0, 2.0],
            [0.0, 2.0, 4.0, 4.0],
            [0.0, 2.0, 4.0, 0.0],
        ]
    )
    y = np.array(
        [
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    expect_x = np.array(
        [
            1.0,
            1.41421356,
            2.0,
            0.0,
            2.82842712,
            4.0,
            0.0,
            2.82842712,
            5.65685425,
            0.0,
        ]
    )
    expect_y = np.array(
        [
            0.0,
            0.0,
            1.0,
            4.24264069,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.41421356,
            0.0,
        ]
    )
    np.testing.assert_allclose(cpsdppy.linalg.svec(x), expect_x)
    np.testing.assert_allclose(
        cpsdppy.linalg.svec(scipy.sparse.csr_array(x)).toarray(),
        expect_x.reshape(-1, 1),
    )
    np.testing.assert_allclose(cpsdppy.linalg.svec(y), expect_y)
    np.testing.assert_allclose(
        cpsdppy.linalg.svec(scipy.sparse.csr_array(y)).toarray(),
        expect_y.reshape(-1, 1),
    )

    # TODO Add batched version? Consider how to batch sparse one.
    # xy = np.stack([x, y])
    # expect_xy = np.stack([expect_x, expect_y])
    # scipy.sparse.csr_array(xy)
    # np.testing.assert_allclose(cpsdppy.linalg.svec(xy), expect_xy)
    # np.testing.assert_allclose(
    #     cpsdppy.linalg.svec(scipy.sparse.csr_array(xy)).toarray(),
    #     expect_xy,
    # )


if __name__ == "__main__":
    test_svec_numpy()

# vimquickrun: pytest
