# -*- coding: utf-8 -*-

"""CPLEX Interface"""

import enum
import tempfile
import typing

import cplex  # type: ignore
import numpy as np
import scipy.sparse  # type: ignore

from cpsdppy.mip_solvers import base


class CplexInterface(base.BaseSolverInterface):
    """Interface to use Cplex

    Examples
    --------
    >>> m = CplexInterface()
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
    [ 0.75  0.5  -0.  ]
    >>> print(m.get_linear_constraint_slacks())
    [ 0.  0. -1.]

    >>> m.remove_linear_constraints(1)
    >>> m.remove_variables(2)
    >>> m.solve()
    >>> print(m.get_solution())
    [0. 2.]
    """

    def __init__(self, config=None) -> None:
        """Initialise a CplexInterface instance"""
        super().__init__(config)
        self.model = cplex.Cplex()
        self.model.set_error_stream(None)
        self.model.set_log_stream(None)
        self.model.set_results_stream(None)
        self.model.set_warning_stream(None)
        self.set_threads(1)

    def set_threads(self, v) -> None:
        """Set the number of threads to be used"""
        self.model.parameters.threads.set(v)

    def get_threads(self) -> int:
        """Get the number of threads to be used

        Examples
        --------
        >>> m = CplexInterface()
        >>> m.set_threads(3)
        >>> m.get_threads()
        3
        """
        return self.model.parameters.threads.get()

    def set_time_limit(self, v) -> None:
        """Set time limit"""
        self.model.parameters.timelimit.set(v)

    def get_time_limit(self) -> float:
        """Get time limit

        Examples
        --------
        >>> m = CplexInterface()
        >>> m.set_time_limit(4)
        >>> m.get_time_limit()
        4.0
        """
        out = self.model.parameters.timelimit.get()
        if out >= 1e75:
            return np.inf
        else:
            return out

    def set_method(self, x) -> None:
        """Set the optimization method"""

        if isinstance(x, str):
            x = Method[x.upper()]
        self.model.parameters.lpmethod.set(x)
        self.model.parameters.qpmethod.set(x)

    def get_method(self):
        """Get the optimization method

        Examples
        --------
        >>> m = CplexInterface()
        >>> m.get_method()
        <Method.AUTOMATIC: 0>
        >>> m.set_method("dual_simplex")
        >>> m.get_method()
        <Method.DUAL_SIMPLEX: 2>
        """
        return Method(self.model.parameters.lpmethod.get())

    def find_unbounded_ray(self, v: bool = True) -> None:
        raise NotImplementedError

    def find_quadratic_constraint_dual(self, v: bool = True) -> None:
        raise NotImplementedError

    def get_n_variables(self) -> int:
        """Get the number of variables

        Examples
        --------
        >>> m = CplexInterface()
        >>> index = m.add_variables(shape=3)
        >>> m.get_n_variables()
        3
        """
        return self.model.variables.get_num()

    def write(self, file_name):
        self.model.write(file_name)

    def write_string(self, format="lp"):
        r"""Create a string containing problem data

        Examples
        --------
        >>> m = CplexInterface()
        >>> m.add_variables(obj=[1, 2, 3], name=["x", "y", "z"])
        array([0, 1, 2])
        >>> m.add_linear_constraints(coef=[2, 3, 4], name="c")
        array([0])
        >>> print(m.write_string())
        \ENCODING=ISO-8859-1
        \Problem name:
        <BLANKLINE>
        Minimize
         obj1: x + 2 y + 3 z
        Subject To
         c: 2 x + 3 y + 4 z  = 0
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
        >>> m = CplexInterface()
        >>> m.add_variables(obj=[1, 2, 3], name=["x", "y", "z"])
        array([0, 1, 2])
        >>> m.add_linear_constraints(coef=[2, 3, 4], name="c")
        array([0])
        >>> m.print()
        \ENCODING=ISO-8859-1
        \Problem name:
        <BLANKLINE>
        Minimize
         obj1: x + 2 y + 3 z
        Subject To
         c: 2 x + 3 y + 4 z  = 0
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
        m = cls(config=config)
        m.model.read(file_path)
        return m

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
        >>> m = CplexInterface.read_string(text)
        >>> m.get_n_variables()
        3
        """
        if format not in ["lp", "mip", "sav"]:
            raise ValueError(f"unknown format '{format}'")

        with tempfile.NamedTemporaryFile("w", suffix=f".{format}") as f:
            f.write(text)
            f.flush()
            out = cls.read(f.name)
        out.model.set_problem_name("")
        return out

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
        >>> m = CplexInterface()
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
        if np.all(type == b"C"[0]):
            type = None
        kwargs = dict(
            lb=lb.astype(float).ravel(),
            ub=ub.astype(float).ravel(),
            obj=obj.astype(float).ravel(),
        )
        if type is not None:
            kwargs["types"] = tostr(type.ravel())
        if name is not None:
            kwargs["names"] = name
        index = self.model.variables.add(**kwargs)
        return np.array(index).reshape(lb.shape)

    def _remove_variables_impl(self, index):
        index = np.atleast_1d(index)
        self.model.variables.delete(index.astype(int).tolist())

    def add_2x2_psd_variables(
        self, n: typing.Optional[int] = None
    ) -> typing.Any:
        """Add a 2 by 2 PSD matrix variable

        Examples
        --------
        >>> m = CplexInterface()
        >>> _ = m.add_2x2_psd_variables()
        >>> m.set_linear_objective_coefs(value=[0, 1, 0])
        >>> _ = m.add_linear_constraints(rhs=1, sense="E", coef=[1, 1, 1])
        >>> m.solve()
        >>> print(np.round(m.get_solution(), 2))
        [ 1. -1.  1.]
        """
        if (n is None) or (n <= 0):
            n = 1
        lb = np.array([[0, -np.inf, 0]]).repeat(n, axis=0)
        variable_indices = self.add_variables(lb=lb)
        qconstr_indices = []
        for i in range(n):
            quad_expr = [
                [int(variable_indices[i, 0]), int(variable_indices[i, 1])],
                [int(variable_indices[i, 2]), int(variable_indices[i, 1])],
                [1.0, -1.0],
            ]
            qconstr_indices.append(
                self.model.quadratic_constraints.add(
                    quad_expr=quad_expr,
                    sense="G",
                    rhs=0.0,
                )
            )
        return variable_indices, np.array(qconstr_indices)

    def set_variable_lb(self, index=None, value=None) -> None:
        """Set variable lower bounds

        Examples
        --------
        >>> m = CplexInterface()
        >>> index = m.add_variables(shape=3)
        >>> print(m.get_variable_lb())
        [0. 0. 0.]
        >>> m.set_variable_lb(index=[0, 2], value=[1, 2])
        >>> print(m.get_variable_lb())
        [1. 0. 2.]
        """
        if index is None:
            variables = np.arange(self.get_n_variables())
        else:
            variables = index
        variables, value = np.broadcast_arrays(variables, value)
        self.model.variables.set_lower_bounds(
            zip(variables.tolist(), value.tolist())
        )

    def set_variable_ub(self, index=None, value=None) -> None:
        """Set variable lower bounds

        Examples
        --------
        >>> m = CplexInterface()
        >>> index = m.add_variables(shape=3)
        >>> print(m.get_variable_ub())
        [inf inf inf]
        >>> m.set_variable_ub(index=[0, 2], value=[1, 2])
        >>> print(m.get_variable_ub())
        [ 1. inf  2.]
        """
        if index is None:
            variables = np.arange(self.get_n_variables())
        else:
            variables = index
        variables, value = np.broadcast_arrays(variables, value)
        self.model.variables.set_upper_bounds(
            zip(variables.tolist(), value.tolist())
        )

    def set_variable_type(self, index, value) -> None:
        """Set variable types

        Examples
        --------
        >>> m = CplexInterface()
        >>> index = m.add_variables(shape=3)
        >>> print(tostr(m.get_variable_type()))
        CCC
        >>> m.set_variable_type(index=[0, 2], value=['I', 'B'])
        >>> print(tostr(m.get_variable_type()))
        ICB
        """
        if index is None:
            variables = np.arange(self.get_n_variables())
        else:
            variables = np.asarray(index, dtype=int)
        value = tobytearray(value)
        variables, value = np.broadcast_arrays(variables, value)
        value = tostr(value.ravel())
        self.model.variables.set_types(zip(variables.tolist(), value))

    def get_variable_lb(self, index=None) -> np.ndarray:
        if index is None:
            out = np.array(self.model.variables.get_lower_bounds())
        else:
            out = np.array(
                self.model.variables.get_lower_bounds(
                    np.asarray(index, dtype=int).tolist()
                )
            )
        out[out <= -1e10] = -np.inf
        out[out >= 1e10] = np.inf
        return out

    def get_variable_ub(self, index=None) -> np.ndarray:
        if index is None:
            out = np.array(self.model.variables.get_upper_bounds())
        else:
            out = np.array(
                self.model.variables.get_upper_bounds(
                    np.asarray(index, dtype=int).tolist()
                )
            )
        out[out <= -1e10] = -np.inf
        out[out >= 1e10] = np.inf
        return out

    def get_variable_type(self, index=None) -> np.ndarray:
        try:
            if index is None:
                out = np.array(self.model.variables.get_types())
            else:
                out = np.array(
                    self.model.variables.get_types(
                        np.asarray(index, dtype=int).tolist()
                    )
                )
            return tobytearray(out)
        except cplex.exceptions.errors.CplexSolverError:
            pass
        if index is None:
            out = np.full(self.get_n_variables(), b"C"[0])
        else:
            out = np.full(index.shape, b"C"[0])
        return out

    def get_n_linear_constraints(self) -> int:
        return self.model.linear_constraints.get_num()

    def get_n_quadratic_constraints(self) -> int:
        return self.model.quadratic_constraints.get_num()

    def add_linear_constraints(
        self, shape=None, sense="E", rhs=0.0, coef=None, name=None
    ) -> np.ndarray:
        """Add linear constraints"""
        sense = tobytearray(sense)
        if shape is not None:
            sense = np.broadcast_to(sense, shape)
            rhs = np.broadcast_to(rhs, shape)
        else:
            sense, rhs = np.broadcast_arrays(sense, rhs)
        kwargs = dict(
            senses=tostr(sense.ravel()), rhs=rhs.astype(float).ravel()
        )
        if name is not None:
            if isinstance(name, str):
                name = [name]
            kwargs["names"] = name
        index = self.model.linear_constraints.add(**kwargs)
        index = np.array(index).reshape(rhs.shape)
        if coef is not None:
            self.set_linear_constraint_coefs_by_matrix(row=index, coef=coef)
        return index

    def _remove_linear_constraints_impl(self, index: np.ndarray) -> None:
        index = np.atleast_1d(index)
        self.model.linear_constraints.delete(index.astype(int).tolist())

    def _remove_quadratic_constraints_impl(self, index: np.ndarray) -> None:
        index = np.atleast_1d(index)
        self.model.quadratic_constraints.delete(index.astype(int).tolist())

    def set_linear_constraint_sense(self, index, value) -> None:
        """Set linear constraint sense

        Examples
        --------
        >>> m = CplexInterface()
        >>> index = m.add_linear_constraints(shape=3)
        >>> print(tostr(m.get_linear_constraint_sense()))
        EEE
        >>> m.set_linear_constraint_sense(index=[0, 2], value=['G', 'L'])
        >>> print(tostr(m.get_linear_constraint_sense()))
        GEL
        """
        if index is None:
            constraints = np.arange(self.get_n_linear_constraints())
        else:
            constraints = np.asarray(index, dtype=int)
        value = tobytearray(value)
        constraints, value = np.broadcast_arrays(constraints, value)
        _value = tostr(value.ravel())
        self.model.linear_constraints.set_senses(
            zip(constraints.ravel().tolist(), _value)
        )

    def set_linear_constraint_rhs(self, index, value) -> None:
        """Set RHS of linear constraints

        Examples
        --------
        >>> m = CplexInterface()
        >>> index = m.add_linear_constraints(shape=3)
        >>> print(m.get_linear_constraint_rhs())
        [0. 0. 0.]
        >>> m.set_linear_constraint_rhs(index=[0, 2], value=[1, 2])
        >>> print(m.get_linear_constraint_rhs())
        [1. 0. 2.]
        """
        if index is None:
            constraints = np.arange(self.get_n_linear_constraints())
        else:
            constraints = index
        constraints, value = np.broadcast_arrays(constraints, value)
        self.model.linear_constraints.set_rhs(
            zip(constraints.tolist(), value.tolist())
        )

    def set_linear_constraint_coefs(self, coefs=None) -> None:
        self.model.linear_constraints.set_coefficients(coefs)

    def set_linear_constraint_coefs_by_matrix(self, row, coef) -> None:
        if isinstance(coef, scipy.sparse.spmatrix):
            coef = coef.tocoo()
            _row = coef.row
            col = coef.col
            value = coef.data
        else:
            coef = np.atleast_2d(coef).astype(float)
            n_new_cuts, n_variables = coef.shape
            _row = np.repeat(np.arange(n_new_cuts), n_variables)
            col = np.tile(np.arange(n_variables), n_new_cuts)
            value = coef.ravel()
        self.set_linear_constraint_coefs(
            zip(row[_row].tolist(), col.tolist(), value)
        )

    def get_linear_constraint_coefs(self, index=None) -> np.ndarray:
        return get_A(self.model, row=index)

    def get_linear_constraint_sense(self, index=None) -> np.ndarray:
        if index is None:
            out = np.array(self.model.linear_constraints.get_senses())
        else:
            out = np.array(
                self.model.linear_constraints.get_senses(
                    np.asarray(index, dtype=int).tolist()
                )
            )
        return tobytearray(out)

    def get_linear_constraint_rhs(self, index=None) -> np.ndarray:
        if index is None:
            out = np.array(self.model.linear_constraints.get_rhs())
        else:
            out = np.array(
                self.model.linear_constraints.get_rhs(
                    np.asarray(index, dtype=int).tolist()
                )
            )
        return out

    def set_objective_sense(self, v) -> None:
        """Set the objective sense

        Examples
        --------
        >>> m = CplexInterface()
        >>> print(m.get_objective_sense())
        min
        >>> m.set_objective_sense("max")
        >>> print(m.get_objective_sense())
        max
        """
        if v in ["max", max]:
            self.model.objective.set_sense(self.model.objective.sense.maximize)
        elif v in ["min", min]:
            self.model.objective.set_sense(self.model.objective.sense.minimize)
        else:
            raise ValueError(v)

    def get_objective_sense(self) -> str:
        sense = self.model.objective.get_sense()
        sense = self.model.objective.sense[self.model.objective.get_sense()]
        return sense[:3]

    def set_linear_objective_coefs(self, index=None, value=None) -> None:
        if index is None:
            variables = np.arange(self.get_n_variables())
        else:
            variables = np.asarray(index).astype(int)
        variables, value = np.broadcast_arrays(variables, value)
        self.model.objective.set_linear(
            zip(map(int, variables.ravel()), map(float, value.ravel()))
        )

    def get_linear_objective_coefs(self, index=None) -> np.ndarray:
        if index is None:
            out = np.array(self.model.objective.get_linear())
        else:
            out = np.array(
                self.model.objective.get_linear(
                    np.asarray(index, dtype=int).tolist()
                )
            )
        return out

    def _solve_impl(self) -> None:
        self.model.solve()

    def get_status(self) -> int:
        return self.model.solution.get_status()

    def is_optimal(self, suboptimal=False) -> bool:
        status = self.model.solution.get_status()
        if status in [1, 101, 102]:
            return True
        if suboptimal and (status in [5, 115]):
            return True
        return False

    def assert_optimal(self, *args, **kwargs) -> None:
        if not self.is_optimal(*args, **kwargs):
            raise ValueError(f"model.status={self.get_status_name()}")

    def get_status_name(self) -> str:
        return self.model.solution.get_status_string()

    def get_objective_value(self) -> float:
        """Get the optimal objective value"""
        return self.model.solution.get_objective_value()

    def get_solution(self) -> np.ndarray:
        return np.array(self.model.solution.get_values())

    def get_unbounded_ray(self) -> np.ndarray:
        raise NotImplementedError

    def get_linear_constraint_dual(self) -> np.ndarray:
        return np.array(self.model.solution.get_dual_values())

    def get_linear_constraint_slacks(self) -> np.ndarray:
        return np.array(self.model.solution.get_linear_slacks())


solver_status_code_to_status_name = {
    1: "CPX_STAT_OPTIMAL",
    2: "CPX_STAT_UNBOUNDED",
    3: "CPX_STAT_INFEASIBLE",
    4: "CPX_STAT_INForUNBD",
    5: "CPX_STAT_OPTIMAL_INFEAS",
    6: "CPX_STAT_NUM_BEST",
    10: "CPX_STAT_ABORT_IT_LIM",
    11: "CPX_STAT_ABORT_TIME_LIM",
    12: "CPX_STAT_ABORT_OBJ_LIM",
    13: "CPX_STAT_ABORT_USER",
    14: "CPX_STAT_FEASIBLE_RELAXED_SUM",
    15: "CPX_STAT_OPTIMAL_RELAXED_SUM",
    16: "CPX_STAT_FEASIBLE_RELAXED_INF",
    17: "CPX_STAT_OPTIMAL_RELAXED_INF",
    18: "CPX_STAT_FEASIBLE_RELAXED_QUAD",
    19: "CPX_STAT_OPTIMAL_RELAXED_QUAD",
    20: "CPX_STAT_OPTIMAL_FACE_UNBOUNDED",
    21: "CPX_STAT_ABORT_PRIM_OBJ_LIM",
    22: "CPX_STAT_ABORT_DUAL_OBJ_LIM",
    23: "CPX_STAT_FEASIBLE",
    24: "CPX_STAT_FIRSTORDER",
    25: "CPX_STAT_ABORT_DETTIME_LIM",
    30: "CPX_STAT_CONFLICT_FEASIBLE",
    31: "CPX_STAT_CONFLICT_MINIMAL",
    32: "CPX_STAT_CONFLICT_ABORT_CONTRADICTION",
    33: "CPX_STAT_CONFLICT_ABORT_TIME_LIM",
    34: "CPX_STAT_CONFLICT_ABORT_IT_LIM",
    35: "CPX_STAT_CONFLICT_ABORT_NODE_LIM",
    36: "CPX_STAT_CONFLICT_ABORT_OBJ_LIM",
    37: "CPX_STAT_CONFLICT_ABORT_MEM_LIM",
    38: "CPX_STAT_CONFLICT_ABORT_USER",
    39: "CPX_STAT_CONFLICT_ABORT_DETTIME_LIM",
    41: "CPX_STAT_BENDERS_NUM_BEST",
    101: "CPXMIP_OPTIMAL",
    102: "CPXMIP_OPTIMAL_TOL",
    103: "CPXMIP_INFEASIBLE",
    104: "CPXMIP_SOL_LIM",
    105: "CPXMIP_NODE_LIM_FEAS",
    106: "CPXMIP_NODE_LIM_INFEAS",
    107: "CPXMIP_TIME_LIM_FEAS",
    108: "CPXMIP_TIME_LIM_INFEAS",
    109: "CPXMIP_FAIL_FEAS",
    110: "CPXMIP_FAIL_INFEAS",
    111: "CPXMIP_MEM_LIM_FEAS",
    112: "CPXMIP_MEM_LIM_INFEAS",
    113: "CPXMIP_ABORT_FEAS",
    114: "CPXMIP_ABORT_INFEAS",
    115: "CPXMIP_OPTIMAL_INFEAS",
    116: "CPXMIP_FAIL_FEAS_NO_TREE",
    117: "CPXMIP_FAIL_INFEAS_NO_TREE",
    118: "CPXMIP_UNBOUNDED",
    119: "CPXMIP_INForUNBD",
    120: "CPXMIP_FEASIBLE_RELAXED_SUM",
    121: "CPXMIP_OPTIMAL_RELAXED_SUM",
    122: "CPXMIP_FEASIBLE_RELAXED_INF",
    123: "CPXMIP_OPTIMAL_RELAXED_INF",
    124: "CPXMIP_FEASIBLE_RELAXED_QUAD",
    125: "CPXMIP_OPTIMAL_RELAXED_QUAD",
    126: "CPXMIP_ABORT_RELAXED",
    127: "CPXMIP_FEASIBLE",
    128: "CPXMIP_POPULATESOL_LIM",
    129: "CPXMIP_OPTIMAL_POPULATED",
    130: "CPXMIP_OPTIMAL_POPULATED_TOL",
    131: "CPXMIP_DETTIME_LIM_FEAS",
    132: "CPXMIP_DETTIME_LIM_INFEAS",
    133: "CPXMIP_ABORT_RELAXATION_UNBOUNDED",
    301: "CPX_STAT_MULTIOBJ_OPTIMAL",
    302: "CPX_STAT_MULTIOBJ_INFEASIBLE",
    303: "CPX_STAT_MULTIOBJ_INForUNBD",
    304: "CPX_STAT_MULTIOBJ_UNBOUNDED",
    305: "CPX_STAT_MULTIOBJ_NON_OPTIMAL",
    306: "CPX_STAT_MULTIOBJ_STOPPED",
}

solver_status_name_to_status_code = {
    v: k for k, v in solver_status_code_to_status_name.items()
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
    """Convert 1D array of bytes to a string

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([b"E"[0], b"F"[0], b"Y"[0], b"Q"[0]])
    >>> tostr(a)
    'EFYQ'
    """
    if isinstance(a, str):
        return a
    elif isinstance(a, bytes):
        a.decode("utf8")
    a = np.asarray(a)
    if "U" in a.dtype.str:
        return "".join(a)
    else:
        return bytes(a.tolist()).decode("utf8")


m = cplex.Cplex()


class Method(enum.IntEnum):
    """Algorithm used to solve continuous models"""

    AUTOMATIC = m.parameters.lpmethod.values.auto
    PRIMAL_SIMPLEX = m.parameters.lpmethod.values.primal
    DUAL_SIMPLEX = m.parameters.lpmethod.values.dual
    BARRIER = m.parameters.lpmethod.values.barrier


def get_A(model: typing.Any, row=None, column=None):
    """Get the coefficient of constraints as a CSR matrix.

    Examples
    --------
    >>> m = cplex.Cplex()
    >>> m.variables.add(lb=[0] * 3, types=['B'] * 3)
    range(0, 3)
    >>> lin_expr = [[[0], [1]], [[0, 1], [2, 3]], [[0, 1, 2], [4, 5, 6]]]
    >>> m.linear_constraints.add(lin_expr=lin_expr)
    range(0, 3)
    >>> get_A(m).toarray()
    array([[1., 0., 0.],
           [2., 3., 0.],
           [4., 5., 6.]])

    Parameters
    ----------
    model : cplex.Cplex

    Returns
    -------
    A : scipy.sparse.csr_matrix
    """
    import scipy.sparse as sparse

    A_row = model.linear_constraints.get_rows()
    if len(A_row) == 0:
        return sparse.csr_matrix((0, 0))
    A_data = np.concatenate([r.val for r in A_row])
    A_indices = np.concatenate([r.ind for r in A_row])
    A_indptr = np.r_[0, np.cumsum([len(r.val) for r in A_row])]
    A = sparse.csr_matrix(
        (A_data, A_indices, A_indptr),
        shape=(
            model.linear_constraints.get_num(),
            model.variables.get_num(),
        ),
    )
    if row is None and column is None:
        return A
    elif row is None:
        return A[:, column]
    elif column is None:
        return A[row, :]
    else:
        return A[np.ix_(row, column)]


def get_Q(model: typing.Any, row1=None, row2=None):
    """Get the quadratic coefficient of the objective as a CSR matrix.

    Examples
    --------
    >>> m = cplex.Cplex()
    >>> m.variables.add(lb=[0] * 3, types=['B'] * 3)
    range(0, 3)
    >>> coefficients = [(0, 0, 1), (1, 0, 2), (1, 2, 3), (2, 2, 4)]
    >>> m.objective.set_quadratic_coefficients(coefficients)
    >>> get_Q(m).toarray()
    array([[1., 2., 0.],
           [2., 0., 3.],
           [0., 3., 4.]])

    Parameters
    ----------
    model : cplex.Cplex

    Returns
    -------
    Q : scipy.sparse.csr_matrix
    """
    import scipy.sparse as sparse

    quad = model.objective.get_quadratic()
    row = []
    col = []
    data = []
    for i, q in enumerate(quad):
        if len(q.ind) == 0:
            continue
        row += [i] * len(q.ind)
        col += q.ind
        data += q.val
    Q = sparse.csr_matrix(
        (data, (row, col)),
        shape=(
            model.variables.get_num(),
            model.variables.get_num(),
        ),
    )
    if row1 is None and row2 is None:
        return Q
    elif row1 is None:
        return Q[:, row2]
    elif row2 is None:
        return Q[row1, :]
    else:
        return Q[np.ix_(row1, row2)]


if __name__ == "__main__":
    import doctest

    n_failuers, _ = doctest.testmod(
        optionflags=doctest.ELLIPSIS + doctest.NORMALIZE_WHITESPACE
    )
    if n_failuers > 0:
        raise ValueError(f"failed {n_failuers} tests")
