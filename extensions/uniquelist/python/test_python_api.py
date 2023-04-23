# -*- coding: utf-8 -*-

"""Test the Python interface of uniquelist.

This runs quick tests on the Python interface of uniquelist.
To run this script, issue the following command in the top directory.

$ python python/test_python_api.py

This should print "all test passed".
"""

import numpy as np
from python_api import uniquelist


def main():
    # ------------------------------------------
    # Test unique_array_list.
    # ------------------------------------------

    list = uniquelist.create_batched_unique_array_list(3, 3)
    pos, new = list.append(
        list_index=[0, 1, 2, 0],
        value=[
            [1, 2, 3],
            [2, 3, 4],
            [1, 2, 5],
            [1, 2, 3],
        ],
    )
    np.testing.assert_equal(pos, [0, 0, 0, 0])
    np.testing.assert_equal(new, [1, 1, 1, 0])
    np.testing.assert_equal(list.size, [1, 1, 1])

    # 0:
    # [1, 2, 3],
    # 1:
    # [2, 3, 4],
    # 2:
    # [1, 2, 5],

    pos, new = list.append(
        list_index=[0, 1, 2],
        value=[
            [3, 4, 5],
            [2, 3, 4],
            [0, 1, 0],
        ],
    )
    np.testing.assert_equal(pos, [1, 0, 1])
    np.testing.assert_equal(new, [1, 0, 1])
    np.testing.assert_equal(list.size, [2, 1, 2])

    # 0:
    # [1, 2, 3],
    # [3, 4, 5],
    # 1:
    # [2, 3, 4],
    # 2:
    # [1, 2, 5],
    # [0, 1, 0],

    pos, new = list.append(
        list_index=[0, 0, 2],
        value=[
            [5, 7, 8],
            [1, 4, 3],
            [0, 0.9999999, 0],
        ],
    )
    np.testing.assert_equal(pos, [2, 3, 1])
    np.testing.assert_equal(new, [1, 1, 0])
    np.testing.assert_equal(list.size, [4, 1, 2])

    # 0:
    # [1, 2, 3],
    # [3, 4, 5],
    # [5, 7, 8],
    # [1, 4, 3],
    # 1:
    # [2, 3, 4],
    # 2:
    # [1, 2, 5],
    # [0, 1, 0],

    list.erase_nonzero(list_index=0, flag=[True, False, False, True])
    np.testing.assert_equal(list.size, [2, 1, 2])
    pos, new = list.append(
        list_index=0,
        value=[3, 4, 5],
    )
    np.testing.assert_equal(pos, 0)
    np.testing.assert_equal(new, 0)
    np.testing.assert_equal(list.size, [2, 1, 2])

    # 0:
    # [3, 4, 5],
    # [5, 7, 8],
    # 1:
    # [2, 3, 4],
    # 2:
    # [1, 2, 5],
    # [0, 1, 0],

    pos, new = list.append(
        list_index=0,
        value=[5, 6, 3],
    )
    np.testing.assert_equal(pos, 2)
    np.testing.assert_equal(new, 1)
    np.testing.assert_equal(list.size, [3, 1, 2])

    # 0:
    # [3, 4, 5],
    # [5, 7, 8],
    # [5, 6, 3],
    # 1:
    # [2, 3, 4],
    # 2:
    # [1, 2, 5],
    # [0, 1, 0],

    res = list.isin(
        [0, 0, 1, 1],
        [
            [1, 2, 3],
            [5, 7, 8],
            [2, 3, 4],
            [5, 7, 8],
        ],
    )

    np.testing.assert_equal(res, [False, True, True, False])

    print("all test passed")


if __name__ == "__main__":
    main()
