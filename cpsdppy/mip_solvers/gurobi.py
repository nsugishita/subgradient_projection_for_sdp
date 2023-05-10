# -*- coding: utf-8 -*-

"""Gurobi Interface"""

import enum
import tempfile
import typing

import gurobipy  # type: ignore
import numpy as np
import scipy.sparse  # type: ignore

from cpsdppy.mip_solvers import base


class GurobiInterface(base.BaseSolverInterface):
    """Interface to use Gurobi

    Examples
    --------
    >>> m = GurobiInterface()
    >>> _ = m.add_variables(shape=3)
    >>> m.set_linear_objective_coefs(value=[1, 2, 1])
    >>> coef = [
    ...    [0, 2, 1],
    ...    [2, 1, 0],
    ...    [1, 2, 1],
    ... ]
    >>> _ = m.add_linear_constraints(rhs=[4, 4, 4], coef=coef, sense="G")
    >>> m.solve()
    >>> m.get_objective_value()
    5.0
    >>> print(m.get_solution())
    [1. 2. 0.]
    >>> print(m.get_linear_constraint_dual())
    [0.75 0.5  0.  ]
    >>> print(m.get_linear_constraint_slacks())
    [ 0.  0. -1.]

    >>> m.remove_linear_constraints(1)
    >>> m.remove_variables(2)
    >>> m.solve()
    >>> print(m.get_solution())
    [0. 2.]
    """

    env = gurobipy.Env(params={"LogToConsole": False})

    def __init__(self, config=None, model=None) -> None:
        """Initialise a GurobiInterface instance"""
        super().__init__(config)
        if model is not None:
            self.model = model
        else:
            self.model = gurobipy.Model(env=self.env)
        self.set_threads(1)

    def set_threads(self, v) -> None:
        """Set number of threads to be used"""
        self.model.setParam(gurobipy.GRB.Param.Threads, v)

    def get_threads(self) -> int:
        return self.model.Params.Threads

    def set_time_limit(self, v) -> None:
        """Set time limit"""
        self.model.setParam(gurobipy.GRB.Param.TimeLimit, max(v, 0))

    def get_time_limit(self) -> float:
        return self.model.Params.TimeLimit

    def set_method(self, x) -> None:
        """Set the optimization method

        Examples
        --------
        >>> m = GurobiInterface()
        >>> m.get_method()
        <Method.AUTOMATIC: -1>
        >>> m.set_method("dual_simplex")
        >>> m.get_method()
        <Method.DUAL_SIMPLEX: 1>
        """
        if isinstance(x, str):
            x = Method[x.upper()]
        self.model.setParam(gurobipy.GRB.Param.Method, x)

    def get_method(self):
        return Method(self.model.Params.Method)

    def find_unbounded_ray(self, v: bool = True) -> None:
        self.model.setParam(gurobipy.GRB.Param.InfUnbdInfo, int(v))

    def find_quadratic_constraint_dual(self, v: bool = True) -> None:
        self.model.setParam(gurobipy.GRB.Param.QCPDual, int(v))

    def get_n_variables(self) -> int:
        """Get the number of variables

        Examples
        --------
        >>> m = GurobiInterface()
        >>> index = m.add_variables(shape=3)
        >>> m.get_n_variables()
        3
        """
        return self.model.NumVars

    def write(self, file_name):
        self.model.write(file_name)

    def write_string(self, format="lp"):
        r"""Create a string containing problem data

        Examples
        --------
        >>> m = GurobiInterface()
        >>> m.add_variables(obj=[1, 2, 3])
        array([0, 1, 2])
        >>> print(m.write_string())
        \ LP format - ...
        Minimize
          C0 + 2 C1 + 3 C2
        Subject To
        Bounds
        End

        Returns
        -------
        res : str
        """
        if format not in ["lp", "mip", "sav"]:
            raise ValueError(f"unknown format '{format}'")
        f = tempfile.NamedTemporaryFile(
            mode="w+", encoding="latin-1", suffix=f".{format}"
        )
        with f:
            self.write(f.name)
            f.seek(0)
            ret = f.read()
        if ret.endswith("\n"):
            ret = ret[:-1]
        return ret

    def print(self, format="lp"):
        r"""Print problem to the console

        Examples
        --------
        >>> m = GurobiInterface()
        >>> m.add_variables(obj=[1, 2, 3])
        array([0, 1, 2])
        >>> m.print()
        \ LP format - ...
        Minimize
          C0 + 2 C1 + 3 C2
        Subject To
        Bounds
        End
        """
        print(self.write_string(format=format))

    @classmethod
    def read(cls, file_path, config={}):
        """Read an MPS, LP or SAV file

        Parameters
        ----------
        file_path : str
        """
        m = gurobipy.read(file_path, env=GurobiInterface.env)
        return GurobiInterface(config=config, model=m)

    @classmethod
    def read_string(cls, text, format="lp"):
        r"""Read problem data from a given text

        Examples
        --------
        >>> text = '''
        ... Minimize
        ... x1 + 2 x2 + 3 x3
        ... Subject To
        ... Bounds
        ... End
        ... '''
        >>> m = GurobiInterface.read_string(text)
        >>> m.get_n_variables()
        3
        """
        if format not in ["lp", "mip", "sav"]:
            raise ValueError(f"unknown format '{format}'")

        with tempfile.NamedTemporaryFile("w", suffix=f".{format}") as f:
            f.write(text)
            f.flush()
            return cls.read(f.name)

    def add_variables(
        self,
        shape=None,
        lb=0.0,
        ub=np.inf,
        obj=0.0,
        type="C",
        name=None,
    ) -> np.ndarray:
        """
        Examples
        --------
        >>> m = GurobiInterface()
        >>> _ = m.add_variables(shape=2)
        >>> m.get_n_variables()
        2
        """
        type = tobytearray(type)
        if shape is not None:
            lb = np.broadcast_to(lb, shape)
            ub = np.broadcast_to(ub, shape)
            obj = np.broadcast_to(obj, shape)
            type = np.broadcast_to(type, shape)
        else:
            lb, ub, obj, type = np.broadcast_arrays(lb, ub, obj, type)
        _type = np.full(type.shape, gurobipy.GRB.CONTINUOUS)
        _type[type == b"B"[0]] = gurobipy.GRB.BINARY
        _type[type == b"I"[0]] = gurobipy.GRB.INTEGER
        n = self.model.NumVars
        self.model.addMVar(
            shape=lb.shape, lb=lb, ub=ub, obj=obj, vtype=_type, name=name
        )
        self.model.update()
        return np.arange(n, n + lb.size).reshape(lb.shape)

    def _remove_variables_impl(self, index):
        index = np.atleast_1d(index)
        self.model.remove(np.array(self.model.getVars())[index].tolist())
        self.model.update()

    def add_2x2_psd_variables(
        self, n: typing.Optional[int] = None
    ) -> typing.Any:
        """Add a 2 by 2 PSD matrix variable

        Examples
        --------
        >>> m = GurobiInterface()
        >>> _ = m.add_2x2_psd_variables()
        >>> m.set_linear_objective_coefs(value=[0, 1, 0])
        >>> _ = m.add_linear_constraints(rhs=1, sense="E", coef=[1, 1, 1])
        >>> m.solve()
        >>> print(np.round(m.get_solution(), 2))
        [ 1. -1.  1.]
        """
        if (n is None) or (n <= 0):
            n = 1
        n_qconstrs = self.model.NumQConstrs
        variable_name = f"v{n_qconstrs}"
        lb = np.array([[0, -np.inf, 0]]).repeat(n, axis=0)
        variable_indices = self.add_variables(
            lb=lb,
            name=variable_name,
        )
        variables = np.array(self.model.getVars())[variable_indices]
        for i in range(n):
            self.model.addConstr(
                variables[i, 0] * variables[i, 2] - variables[i, 1] ** 2 >= 0
            )
        qconstr_indices = np.arange(n_qconstrs, n_qconstrs + n)
        self.model.update()
        return variable_indices, qconstr_indices

    def set_variable_lb(self, index=None, value=None) -> None:
        """Set variable lower bounds

        Examples
        --------
        >>> m = GurobiInterface()
        >>> index = m.add_variables(shape=3)
        >>> print(m.get_variable_lb())
        [0. 0. 0.]
        >>> m.set_variable_lb(index=[0, 2], value=[1, 2])
        >>> print(m.get_variable_lb())
        [1. 0. 2.]
        """
        if index is None:
            variables = self.model.getVars()
        else:
            variables = np.array(self.model.getVars())[index]
        variables, value = np.broadcast_arrays(variables, value)
        self.model.setAttr("LB", variables, value)
        self.model.update()

    def set_variable_ub(self, index=None, value=None) -> None:
        """Set variable lower bounds

        Examples
        --------
        >>> m = GurobiInterface()
        >>> index = m.add_variables(shape=3)
        >>> print(m.get_variable_ub())
        [inf inf inf]
        >>> m.set_variable_ub(index=[0, 2], value=[1, 2])
        >>> print(m.get_variable_ub())
        [ 1. inf  2.]
        """
        if index is None:
            variables = self.model.getVars()
        else:
            variables = np.array(self.model.getVars())[index]
        variables, value = np.broadcast_arrays(variables, value)
        self.model.setAttr("UB", variables, value)
        self.model.update()

    def set_variable_type(self, index, value) -> None:
        """Set variable types

        Examples
        --------
        >>> m = GurobiInterface()
        >>> index = m.add_variables(shape=3)
        >>> print(tostr(m.get_variable_type()))
        CCC
        >>> m.set_variable_type(index=[0, 2], value=['I', 'B'])
        >>> print(tostr(m.get_variable_type()))
        ICB
        """
        if index is None:
            variables = self.model.getVars()
        else:
            variables = np.array(self.model.getVars())[index]
        value = tobytearray(value)
        _value = np.full(value.shape, gurobipy.GRB.CONTINUOUS)
        _value[value == b"B"[0]] = gurobipy.GRB.BINARY
        _value[value == b"I"[0]] = gurobipy.GRB.INTEGER
        variables, _value = np.broadcast_arrays(variables, _value)
        self.model.setAttr("VType", variables, _value)
        self.model.update()

    def get_variable_lb(self, index=None) -> np.ndarray:
        if index is None:
            variables = self.model.getVars()
        else:
            variables = np.array(self.model.getVars())[index]
        return np.array(self.model.getAttr("LB", variables))

    def get_variable_ub(self, index=None) -> np.ndarray:
        if index is None:
            variables = self.model.getVars()
        else:
            variables = np.array(self.model.getVars())[index]
        return np.array(self.model.getAttr("UB", variables))

    def get_variable_type(self, index=None) -> np.ndarray:
        if index is None:
            variables = self.model.getVars()
        else:
            variables = np.array(self.model.getVars())[index]
        type = self.model.getAttr("VType", variables)
        return tobytearray(type)

    def get_n_linear_constraints(self) -> int:
        return self.model.NumConstrs

    def get_n_quadratic_constraints(self) -> int:
        return self.model.NumQConstrs

    def add_linear_constraints(
        self, shape=None, sense="E", rhs=0.0, coef=None, name=None
    ) -> np.ndarray:
        """Add linear constraints"""
        if name is not None:
            raise NotImplementedError
        sense = tobytearray(sense)
        if shape is not None:
            sense = np.broadcast_to(sense, shape)
            rhs = np.broadcast_to(rhs, shape)
        else:
            sense, rhs = np.broadcast_arrays(sense, rhs)
        casted_sense = np.full(sense.shape, gurobipy.GRB.LESS_EQUAL)
        casted_sense[sense == b"E"[0]] = gurobipy.GRB.EQUAL
        casted_sense[sense == b"L"[0]] = gurobipy.GRB.LESS_EQUAL
        casted_sense[sense == b"G"[0]] = gurobipy.GRB.GREATER_EQUAL
        A = scipy.sparse.coo_array(([], ([], [])), shape=(len(rhs), 0))
        n = self.model.NumConstrs
        self.model.addMConstr(A, [], casted_sense, rhs)
        self.model.update()
        new_constraint_indices = np.arange(n, n + sense.size).reshape(
            sense.shape
        )
        if coef is not None:
            self.set_linear_constraint_coefs_by_matrix(
                row=new_constraint_indices, coef=coef
            )
        return new_constraint_indices

    def _remove_linear_constraints_impl(self, index: np.ndarray) -> None:
        index = np.atleast_1d(index)
        self.model.remove(np.array(self.model.getConstrs())[index].tolist())
        self.model.update()

    def _remove_quadratic_constraints_impl(self, index: np.ndarray) -> None:
        index = np.atleast_1d(index)
        self.model.remove(np.array(self.model.getQConstrs())[index].tolist())
        self.model.update()

    def set_linear_constraint_sense(self, index, value) -> None:
        """Set linear constraint sense

        Examples
        --------
        >>> m = GurobiInterface()
        >>> index = m.add_linear_constraints(shape=3)
        >>> print(tostr(m.get_linear_constraint_sense()))
        ===
        >>> m.set_linear_constraint_sense(index=[0, 2], value=['G', 'L'])
        >>> print(tostr(m.get_linear_constraint_sense()))
        >=<
        """
        if index is None:
            constraints = self.model.getConstrs()
        else:
            constraints = np.array(self.model.getConstrs())[index]
        value = tobytearray(value)
        _value = np.full(value.shape, gurobipy.GRB.EQUAL)
        _value[value == b"G"[0]] = gurobipy.GRB.GREATER_EQUAL
        _value[value == b"L"[0]] = gurobipy.GRB.LESS_EQUAL
        constraints, _value = np.broadcast_arrays(constraints, _value)
        self.model.setAttr("Sense", constraints, _value)
        self.model.update()

    def set_linear_constraint_rhs(self, index, value) -> None:
        """Set RHS of linear constraints

        Examples
        --------
        >>> m = GurobiInterface()
        >>> index = m.add_linear_constraints(shape=3)
        >>> print(m.get_linear_constraint_rhs())
        [0. 0. 0.]
        >>> m.set_linear_constraint_rhs(index=[0, 2], value=[1, 2])
        >>> print(m.get_linear_constraint_rhs())
        [1. 0. 2.]
        """
        if index is None:
            constraints = self.model.getConstrs()
        else:
            constraints = np.array(self.model.getConstrs())[index]
        constraints, value = np.broadcast_arrays(constraints, value)
        self.model.setAttr("RHS", constraints, value)
        self.model.update()

    def set_linear_constraint_coefs(self, coefs=None) -> None:
        vars = self.model.getVars()
        cons = self.model.getConstrs()
        for row, col, val in coefs:
            self.model.chgCoeff(cons[row], vars[col], val)
        self.model.update()

    def set_linear_constraint_coefs_by_matrix(self, row, coef) -> None:
        if isinstance(coef, scipy.sparse.spmatrix):
            coef = coef.tocoo()
            _row = coef.row
            col = coef.col
            value = coef.data
        else:
            coef = np.atleast_2d(coef)
            n_new_cuts, n_variables = coef.shape
            _row = np.repeat(np.arange(n_new_cuts), n_variables)
            col = np.tile(np.arange(n_variables), n_new_cuts)
            value = coef.ravel()
        self.set_linear_constraint_coefs(zip(row[_row], col, value))

    def get_linear_constraint_coefs(self, index=None) -> np.ndarray:
        if index is None:
            index = np.arange(self.get_n_linear_constraints())
        else:
            index = np.asarray(index)
        return self.model.getA()[index]

    def get_linear_constraint_sense(self, index=None) -> np.ndarray:
        if index is None:
            constraints = self.model.getConstrs()
        else:
            constraints = np.array(self.model.getConstrs())[index]
        type = self.model.getAttr("sense", constraints)
        return tobytearray(type)

    def get_linear_constraint_rhs(self, index=None) -> np.ndarray:
        if index is None:
            constraints = self.model.getConstrs()
        else:
            constraints = np.array(self.model.getConstrs())[index]
        return np.array(self.model.getAttr("RHS", constraints))

    def set_objective_sense(self, v) -> None:
        """Set the objective sense

        Examples
        --------
        >>> m = GurobiInterface()
        >>> print(m.get_objective_sense())
        min
        >>> m.set_objective_sense("max")
        >>> print(m.get_objective_sense())
        max
        """
        if v in ["max", max]:
            self.model.ModelSense = -1
        elif v in ["min", min]:
            self.model.ModelSense = 1
        else:
            raise ValueError(v)
        self.model.update()

    def get_objective_sense(self) -> str:
        sense = self.model.ModelSense
        if sense == -1:
            return "max"
        elif sense == 1:
            return "min"
        else:
            raise ValueError(f"invalid model sense: {sense}")

    def set_linear_objective_coefs(self, index=None, value=None) -> None:
        if index is None:
            variables = self.model.getVars()
        else:
            variables = np.array(self.model.getVars())[index]
        variables, value = np.broadcast_arrays(variables, value)
        self.model.setAttr(gurobipy.GRB.Attr.Obj, variables, value)

    def get_linear_objective_coefs(self, index=None) -> np.ndarray:
        obj = np.asarray(
            self.model.getAttr(gurobipy.GRB.Attr.Obj, self.model.getVars())
        )
        if index is None:
            return obj
        else:
            return obj[index]

    def _solve_impl(self) -> None:
        self.model.optimize()

    def get_status(self) -> int:
        return self.model.Status

    def is_optimal(self, suboptimal=False) -> bool:
        status = self.model.Status
        if status == 2:
            return True
        if suboptimal and (status == 13):
            return True
        return False

    def assert_optimal(self, *args, **kwargs) -> None:
        if not self.is_optimal(*args, **kwargs):
            raise ValueError(f"model.status={self.get_status_name()}")

    def get_status_name(self) -> str:
        return solver_status_code_to_status_name[self.model.Status]

    def get_objective_value(self) -> float:
        """Get the optimal objective value"""
        return self.model.ObjVal

    def get_solution(self) -> np.ndarray:
        return np.array(
            self.model.getAttr(gurobipy.GRB.Attr.X, self.model.getVars())
        )

    def get_unbounded_ray(self) -> np.ndarray:
        return self.model.getAttr(
            gurobipy.GRB.Attr.UnbdRay, self.model.getVars()
        )

    def get_linear_constraint_dual(self) -> np.ndarray:
        return np.array(
            self.model.getAttr(gurobipy.GRB.Attr.Pi, self.model.getConstrs())
        )

    def get_linear_constraint_slacks(self) -> np.ndarray:
        return np.array(
            self.model.getAttr(
                gurobipy.GRB.Attr.Slack, self.model.getConstrs()
            )
        )


solver_status_code_to_status_name = {
    1: "LOADED",
    2: "OPTIMAL",
    3: "INFEASIBLE",
    4: "INF_OR_UNBD",
    5: "UNBOUNDED",
    6: "CUTOFF",
    7: "ITERATION_LIMIT",
    8: "NODE_LIMIT",
    9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT",
    11: "INTERRUPTED",
    12: "NUMERIC",
    13: "SUBOPTIMAL",
    14: "INPROGRESS",
    15: "USER_OBJ_LIMIT",
    16: "WORK_LIMIT",
}


def tobytearray(a):
    if a is None:
        return a
    if isinstance(a, (list, tuple, np.ndarray)):
        a = np.asarray(a)
        if "U" in a.dtype.str:
            a = "".join(a)
    if isinstance(a, str):
        a = np.array(list(bytes(a, "utf8")))
    return a


def tostr(a):
    if isinstance(a, str):
        return a
    elif isinstance(a, bytes):
        a.decode("utf8")
    a = np.asarray(a)
    if "U" in a.dtype.str:
        return "".join(a)
    else:
        return bytes(a.tolist()).decode("utf8")


class Method(enum.IntEnum):
    """Algorithm used to solve continuous models"""

    AUTOMATIC = -1
    PRIMAL_SIMPLEX = 0
    DUAL_SIMPLEX = 1
    BARRIER = 2
    CONCURRENT = 3
    DETERMINISTIC_CONCURRENT = 4
    DETERMINISTIC_CONCURRENT_SIMPLEX = 5


if __name__ == "__main__":
    import doctest

    n_failuers, _ = doctest.testmod(
        optionflags=doctest.ELLIPSIS + doctest.NORMALIZE_WHITESPACE
    )
    if n_failuers > 0:
        raise ValueError(f"failed {n_failuers} tests")
