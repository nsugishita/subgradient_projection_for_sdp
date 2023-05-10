# -*- coding: utf-8 -*-

"""Test CplexInterface"""

import numpy as np

from cpsdppy.mip_solvers import cplex_interface


def test_threads():
    s = cplex_interface.CplexInterface()
    assert s.get_threads() == 1
    s.set_threads(3)
    assert s.get_threads() == 3


def test_time_limit():
    s = cplex_interface.CplexInterface()
    assert s.get_time_limit() == np.inf
    s.set_time_limit(3)
    assert s.get_time_limit() == 3


def test_add_variables():
    s = cplex_interface.CplexInterface()
    s.add_variables(shape=(2, 3))
    assert s.get_n_variables() == 6


def test_add_2x2_psd_variables():
    s = cplex_interface.CplexInterface()
    s.add_2x2_psd_variables()
    t = cplex_interface.CplexInterface.read_string(
        """
        Minimize
         obj1: 0 x1 + 0 x2 + 0 x3
        Subject To
         q1: [ x1 * x3 - x2 ^2 ] >= 0
        Bounds
              x2 Free
        End
        """
    )
    assert s.write_string() == t.write_string()


def test_remove_variables():
    s = cplex_interface.CplexInterface()
    s.add_variables(shape=(2, 3))
    s.remove_variables([2, 4])
    assert s.get_n_variables() == 4


def test_get_variable_lb():
    s = cplex_interface.CplexInterface()
    s.add_variables(shape=(2, 3))
    s.set_variable_lb(index=[2, 3, 4], value=[1, 2, 3])
    np.testing.assert_allclose(s.get_variable_lb(), [0, 0, 1, 2, 3, 0])
    s.remove_variables([1, 3])
    np.testing.assert_allclose(s.get_variable_lb(), [0, 1, 3, 0])


def test_get_variable_ub():
    s = cplex_interface.CplexInterface()
    s.add_variables(shape=(2, 3))
    s.set_variable_ub(index=[2, 3, 4], value=[1, 2, 3])
    np.testing.assert_allclose(
        s.get_variable_ub(), [np.inf, np.inf, 1, 2, 3, np.inf]
    )
    s.remove_variables([1, 3])
    np.testing.assert_allclose(s.get_variable_ub(), [np.inf, 1, 3, np.inf])


# vimquickrun: pytest
