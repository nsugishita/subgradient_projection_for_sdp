# -*- coding: utf-8 -*-

import indexremove
import numpy as np


def impl(index, removed):
    """Compute index remove in a naive but slow way"""
    return index - np.sum(index[:, None] > removed, axis=1)


def main():
    """Test indexremove package"""
    index = np.array([0, 2, 3, 6, 10, 15])
    removed = np.array([5, 8, 14])
    print(impl(index, removed))
    indexremove.remove(index, removed)
    print(index)


if __name__ == "__main__":
    main()
