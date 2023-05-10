# -*- coding: utf-8 -*-

from cpsdppy.mip_solvers import cplex_interface, gurobi_interface  # noqa: F401


def get_solver_interface(name):
    name = name.lower()
    if name == "gurobi":
        return gurobi_interface.GurobiInterface()
    elif name == "cplex":
        return cplex_interface.CplexInterface()
    else:
        raise ValueError("unknown solver interface: {name}")
