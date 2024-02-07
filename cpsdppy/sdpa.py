# -*- coding: utf-8 -*-

"""Parser to read data in SDPA format

SDPA format is a data format to store a SDP problem of the following form:

(P)    min   c1*x1+c2*x2+...+cm*xm
       s.t.  F1*x1+F2*x2+...+Fm*xm - F0 = X
                                     X >= 0

The dual of the problem is:

(D)    max   tr(F0*Y)
       s.t.  tr(Fi*Y) = ci    i = 1,2,...,m
                   Y >= 0.

Here all of the matrices F0, F1, ..., Fm, X, and Y are assumed to be symmetric
of size n by n. The constraints X >= 0 and Y >= 0 mean that X and Y must be
positive semidefinite.

The data is processed into the following format

min   c^T x
s.t   sum_{j = 1}^m A_{ij} x_j - A_{i0} : PSD   for i = 1, 2, ..., n
      B x >= b,
      x_lb <= x <= x_ub.

name in program                  symbol      description
-------------------------------  ----------  ---------------------------------
n_variables                      n           int
objective_sense                  min max     'min', 'max'
objective_offset                 offset      float
variable_lb                      x_lb        (n_vars,) array of float
variable_ub                      x_ub        (n_vars,) array of float
objective_coefficient            c           (n_vars,) array of float
linear_constraint_coefficient    B           (n_cons, n_vars) COO
linear_constraint_offset         b           (n_cons,) array of float
linear_constraint_sense          <= = >=     (n_cons,) array of 'E', 'G', 'L'
lmi_constraint_coefficient       A_{ij}      (n, m)-list of COO
lmi_constraint_offset            A_{i0}      n-list of COO
lmi_svec_constraint_coefficient              n-list of COO
lmi_svec_constraint_offset                   n-list of 1d array
-------------------------------  ----------  --------------------------------
"""

import collections
import io
import os
import re
import textwrap

import numpy as np
import pandas as pd
import scipy.sparse

from cpsdppy import linalg


def read(problem_name, config=None):
    """Read data of SDPA format

    Examples
    --------
    >>> data = read('control1.dat-s')
    >>> data['n_variables']
    21
    >>> len(data['lmi_constraint_coefficient'])
    2
    """
    if isinstance(problem_name, dict):
        _problem_name = problem_name["problem_name"]
    elif hasattr(problem_name, "problem_name"):
        _problem_name = getattr(problem_name, "problem_name")
    else:
        _problem_name = problem_name
    if _problem_name == "":
        raise ValueError("problem_name is missing")
    if "." not in _problem_name and "\n" not in _problem_name:
        _problem_name = _problem_name + ".dat-s"
    sdpa_data = _read(_problem_name, config)
    return _convert_sdpa_data_to_dict(_problem_name, sdpa_data, config)


def _convert_sdpa_data_to_dict(problem_name, data, config):
    c, F = data
    objective_coefficient = np.asarray(c)
    n_variables = len(objective_coefficient)
    variable_lb = np.full(n_variables, -np.inf)
    variable_ub = np.full(n_variables, np.inf)
    linear_constraint_coefficient = scipy.sparse.csr_array(
        ([], ([], [])), shape=(0, n_variables)
    )
    linear_constraint_sense = np.array([], dtype=int)
    linear_constraint_rhs = np.array([], dtype=float)
    lmi_constraint_coefficient = []
    lmi_constraint_offset = []
    lmi_svec_constraint_coefficient = []
    lmi_svec_constraint_offset = []
    n_blocks = F[0].n_blocks
    for block_index in range(n_blocks):
        smat_F = [x.tocsr(block=block_index) for x in F]
        svec_F = [
            linalg.svec(
                x.tocoo(block=block_index),
            )
            for x in F
        ]
        lmi_constraint_coefficient.append(smat_F[1:])
        lmi_constraint_offset.append(smat_F[0])
        lmi_svec_constraint_coefficient.append(
            scipy.sparse.hstack(svec_F[1:], format="csr")
        )
        lmi_svec_constraint_offset.append(svec_F[0].toarray().reshape(-1))
    return dict(
        target_objective=get_optimal_objective_value(problem_name),
        n_variables=n_variables,
        objective_sense="min",
        objective_offset=0.0,
        variable_lb=variable_lb,
        variable_ub=variable_ub,
        objective_coefficient=objective_coefficient,
        linear_constraint_coefficient=linear_constraint_coefficient,
        linear_constraint_sense=linear_constraint_sense,
        linear_constraint_rhs=linear_constraint_rhs,
        lmi_constraint_coefficient=lmi_constraint_coefficient,
        lmi_constraint_offset=lmi_constraint_offset,
        lmi_svec_constraint_coefficient=lmi_svec_constraint_coefficient,
        lmi_svec_constraint_offset=lmi_svec_constraint_offset,
    )


def _read(file, config):
    """Read data of SDPA format

    This reads data of SDPA format. The data is parsed and the index
    is adjusted so that row and column number are 0-based (i.e. starts
    with 0).

    Examples
    --------
    >> data = '''
    ... " This is sample data of the following problem:
    ... "
    ... "     min   10 * x1 + 20 * x2
    ... " s.t.  X = F1 * x1 + F2 * x2 - F0
    ... "       X >= 0
    ... "
    ... " where
    ... "
    ... " F0 = [1 0 0 0
    ... "       0 2 0 0
    ... "       0 0 3 0
    ... "       0 0 0 4]
    ... "
    ... " F1 = [1 0 0 0
    ... "       0 1 0 0
    ... "       0 0 0 0
    ... "       0 0 0 0]
    ... "
    ... " F2 = [0 0 0 0
    ... "       0 1 0 0
    ... "       0 0 5 2
    ... "       0 0 2 6]
    ... "
    ... 2 =mdim
    ... 2 =nblocks
    ... {2, 2}
    ... 10.0 20.0
    ... 0 1 1 1 1.0
    ... 0 1 2 2 2.0
    ... 0 2 1 1 3.0
    ... 0 2 2 2 4.0
    ... 1 1 1 1 1.0
    ... 1 1 2 2 1.0
    ... 2 1 2 2 1.0
    ... 2 2 1 1 5.0
    ... 2 2 1 2 2.0
    ... 2 2 2 2 6.0
    ... '''
    >> c, F = read(data)
    >> print(c)
    [10. 20.]
    >> print(F[0].row)
    [0 1 0 1]
    >> import pprint
    >> print(F[0].toarray())
    [[1. 0. 0. 0.]
     [0. 2. 0. 0.]
     [0. 0. 3. 0.]
     [0. 0. 0. 4.]]
    >> print(F[1].toarray())
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    >> print(F[2].toarray())
    [[0. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 5. 0.]
     [0. 0. 2. 6.]]

    One can scale off-diagonal entries. It is useful when one use svec later.

    >> config = {'sdpa_data.offdiagonal_scaling': np.sqrt(2)}
    >> c, F = read(data, config)
    >> print(F[2].toarray().round(2))
    [[0.   0.   0.   0.  ]
     [0.   1.   0.   0.  ]
     [0.   0.   5.   0.  ]
     [0.   0.   2.83 6.  ]]

    Parameters
    ----------
    file : str or file-like object
        If this is a str with multiple lines, the content
        of this str is parsed. If this is a str of a single line,
        this is assumed to be a file path. If this is a file-like
        object, the contents are read and parsed.

    Returns
    -------
    c : list of float
    F : list of SDPAMatrix
    """
    if isinstance(file, str) and "\n" in file:
        # Data is given as a text
        f = io.StringIO(textwrap.dedent(file).strip())
        return _read(f, config)
    elif isinstance(file, str):
        file = os.path.expanduser(file)
        file = os.path.splitext(file)[0] + ".dat-s"
        with open(file, "r") as f:
            return _read(f, config)

    if config is None:
        config = {}

    _config = {k: v for k, v in config.items() if k.startswith("sdpa_data")}
    for k, v in read_config.items():
        _config.setdefault(k, v)

    unknown_config_names = set(_config.keys()) - set(read_config.keys())

    if unknown_config_names:
        unknown_configs = {k: _config[k] for k in unknown_config_names}
        raise ValueError(f"unknown configs: {unknown_configs}")

    _part = str(_config["sdpa_data.matrix_part"]).lower()
    part_lower = 0
    part_upper = 1
    part_full = 2
    if _part in ["l", "lower"]:
        part = part_lower
    elif _part in ["u", "upper"]:
        part = part_upper
    elif _part in ["f", "full"]:
        part = part_full
    else:
        raise ValueError(
            "invalid value of sdpa_data.matrix_part: "
            f"{_config['sdpa_data.matrix_part']}"
        )
    offdiagonal_scaling = _config["sdpa_data.offdiagonal_scaling"]

    mode_comment = 0
    mode_m = 1
    mode_n_blocks = 2
    mode_block_sizes = 3
    mode_c = 4
    mode_matrix_data = 5

    mode = mode_comment

    F = []

    for line_number, line in enumerate(file):
        line = line.strip()

        if line == "":
            continue

        if mode == mode_comment:
            if line[0] in '*"':
                continue
            else:
                # This is not a comment line. Try to parse the current
                # line in the next mode.
                mode += 1

        if mode == mode_m:
            try:
                m = _parse_int(line).value
            except ValueError:
                raise ValueError(
                    f"expected m (int) but got {line} in line "
                    f"{line_number}"
                ) from None
            mode += 1
            continue

        if mode == mode_n_blocks:
            try:
                n_blocks = _parse_int(line).value
            except ValueError:
                raise ValueError(
                    f"expected n_blocks (int) but got {line} in line "
                    f"{line_number}"
                ) from None
            mode += 1
            continue

        if mode == mode_block_sizes:
            tokens = re.sub("[(){},]", " ", line).strip().split()
            try:
                block_sizes = [int(x) for x in tokens]
            except ValueError:
                raise ValueError(
                    f"expected block sizes (list of ints) but got {line} "
                    f"in line {line_number}"
                ) from None
            block_sizes = np.array([x if x > 0 else -x for x in block_sizes])
            mode += 1
            F = [SDPAMatrix(n_blocks, block_sizes) for _ in range(m + 1)]
            continue

        if mode == mode_c:
            tokens = re.sub("[(){},]", " ", line).strip().split()
            try:
                c = [float(x) for x in tokens]
            except ValueError:
                raise ValueError(
                    f"expected c (list of floats) but got {line} in line "
                    f"{line_number}"
                ) from None
            c = np.array(c)
            mode += 1
            continue

        if mode == mode_matrix_data:
            parsed = _parse_matrix_data_line(line)

            if part == part_full:
                _F = F[parsed.matrix_number]
                _F.block_number.append(parsed.block_number - 1)
                _F.row.append(parsed.row - 1)
                _F.column.append(parsed.column - 1)
                _F.entry.append(parsed.entry)

                if parsed.row != parsed.column:
                    _F.block_number.append(parsed.block_number - 1)
                    _F.row.append(parsed.column - 1)
                    _F.column.append(parsed.row - 1)
                    _F.entry.append(parsed.entry)

            elif part == part_lower:
                _F = F[parsed.matrix_number]

                if parsed.row == parsed.column:
                    _F.block_number.append(parsed.block_number - 1)
                    _F.row.append(parsed.row - 1)
                    _F.column.append(parsed.column - 1)
                    _F.entry.append(parsed.entry)

                elif parsed.row > parsed.column:
                    _F.block_number.append(parsed.block_number - 1)
                    _F.row.append(parsed.row - 1)
                    _F.column.append(parsed.column - 1)
                    _F.entry.append(parsed.entry * offdiagonal_scaling)

                else:
                    _F.block_number.append(parsed.block_number - 1)
                    _F.row.append(parsed.column - 1)
                    _F.column.append(parsed.row - 1)
                    _F.entry.append(parsed.entry * offdiagonal_scaling)

            else:
                _F = F[parsed.matrix_number]

                if parsed.row == parsed.column:
                    _F.block_number.append(parsed.block_number - 1)
                    _F.column.append(parsed.row - 1)
                    _F.row.append(parsed.column - 1)
                    _F.entry.append(parsed.entry)

                elif parsed.row < parsed.column:
                    _F.block_number.append(parsed.block_number - 1)
                    _F.column.append(parsed.row - 1)
                    _F.row.append(parsed.column - 1)
                    _F.entry.append(parsed.entry * offdiagonal_scaling)

                else:
                    _F.block_number.append(parsed.block_number - 1)
                    _F.column.append(parsed.column - 1)
                    _F.row.append(parsed.row - 1)
                    _F.entry.append(parsed.entry * offdiagonal_scaling)

    for _F in F:
        _F.block_number = np.array(_F.block_number)
        _F.row = np.array(_F.row)
        _F.column = np.array(_F.column)
        _F.entry = np.array(_F.entry)

    return c, F
    # return SDPAData(c, F)


read_config = {
    # The part of matrix the data should be fill in.
    # "L": Lower, "U": Upper, "F": Full.
    "sdpa_data.matrix_part": "F",
    # When sdpa_data.matrix_part is not "F", this scaling
    # factor is multipled to off-diagonal entries.
    "sdpa_data.offdiagonal_scaling": 1.0,
}


class SDPAData:
    """Data structure to hold SDP data"""

    __slots__ = "c F".split()

    def __init__(self, c, F):
        """Initialise a SDPAData instance"""
        self.c = c
        self.F = F

    def __iter__(self):
        yield self.c
        yield self.F


class SDPAMatrix:
    """Data structure to hold matrix data in SDPA sparse format

    Attributes
    ----------
    n_blocks : int
    block_sizes : list of `n_blocks` ints
    block_number : list of int
    row : list of int
    column : list of int
    entry : list of float
    """

    __slots__ = "n_blocks block_sizes block_number row column entry".split()

    def __init__(self, n_blocks, block_sizes):
        """Initialise a SDPAMatrix instance"""
        self.n_blocks = n_blocks
        self.block_sizes = block_sizes
        self.block_number = []
        self.row = []
        self.column = []
        self.entry = []

    def _tolist_or_array(self, asarray=True):
        size = sum(self.block_sizes)
        if asarray:
            res = np.zeros((size, size))
        else:
            res = [[0.0 for i in range(size)] for j in range(size)]
        start_index = []
        start_index_buf = 0
        for x in self.block_sizes:
            start_index.append(start_index_buf)
            start_index_buf += x
        iter = zip(self.block_number, self.row, self.column, self.entry)
        for block_number, row, column, entry in iter:
            _row = start_index[block_number] + row
            _column = start_index[block_number] + column
            res[_row][_column] = entry

        return res

    def tolist(self):
        """Return a list representing the data as a dense matrix

        Returns
        -------
        res : list of list
        """
        return self._tolist_or_array(asarray=False)

    def toarray(self, block=None):
        """Return an array representing the data as a dense matrix

        Returns
        -------
        res : 2d array
        """
        return self.tocoo(block=block).toarray()

    def tocoo(self, block=None):
        import scipy.sparse

        if block is None:
            size = sum(self.block_sizes)
            n = len(self.block_number)
            row_buffer = np.empty(n, dtype=np.int32)
            column_buffer = np.empty(n, dtype=np.int32)
            data_buffer = np.empty(n, dtype=float)
            start_index = []
            start_index_buf = 0
            for x in self.block_sizes:
                start_index.append(start_index_buf)
                start_index_buf += x
            iter = zip(self.block_number, self.row, self.column, self.entry)
            for i, (block_number, row, column, entry) in enumerate(iter):
                row_buffer[i] = start_index[block_number] + row
                column_buffer[i] = start_index[block_number] + column
                data_buffer[i] = entry
            return scipy.sparse.coo_matrix(
                (data_buffer, (row_buffer, column_buffer)),
                shape=(size, size),
            )
        else:
            size = self.block_sizes[block]
            selector = np.nonzero(self.block_number == block)[0]
            n = selector.size
            row_buffer = np.empty(n, dtype=np.int32)
            column_buffer = np.empty(n, dtype=np.int32)
            data_buffer = np.empty(n, dtype=float)
            iter = zip(
                self.row[selector], self.column[selector], self.entry[selector]
            )
            for i, (row, column, entry) in enumerate(iter):
                row_buffer[i] = row
                column_buffer[i] = column
                data_buffer[i] = entry
            return scipy.sparse.coo_matrix(
                (data_buffer, (row_buffer, column_buffer)),
                shape=(size, size),
            )

    def tocsr(self, block=None):
        return self.tocoo(block=block).tocsr()


def _parse_matrix_data_line(line):
    """Parse a single line of matrix data in SDPA sparse format

    This parses a single line of matrix data in SDPA sparse format.
    Note that this does not adjust the index (SDPA used 1-based
    index).

    Examples
    --------
    >>> _parse_matrix_data_line("4 3 2 3 -4.3")
    matrix_data(matrix_number=4, block_number=3, row=2, column=3, entry=-4.3)

    Parameters
    ----------
    line : str

    Returns
    -------
    matrix_number : int
    block_number : int
    row : int
    column : int
    entry : float
    """
    tokens = line.split()
    if len(tokens) != 5:
        raise ValueError
    return matrix_data(
        int(tokens[0]),
        int(tokens[1]),
        int(tokens[2]),
        int(tokens[3]),
        float(tokens[4]),
    )


matrix_data = collections.namedtuple(
    "matrix_data", "matrix_number block_number row column entry"
)


def _parse_int(text, pos=0):
    """Parse int from a text

    This reads as many characters as possible and parse it as int.

    Examples
    --------
    >>> _parse_int("1")
    parsed(value=1, pos=1)
    >>> _parse_int(" -43 =value")
    parsed(value=-43, pos=5)
    >>> _parse_int(" a-43 =value")
    Traceback (most recent call last):
    ...
    ValueError
    >>> _parse_int("spam")
    Traceback (most recent call last):
    ...
    ValueError

    Parameters
    ----------
    text : str
    pos : int, default 0
        From which the parse should start

    Returns
    -------
    value : int
    pos : int
    """
    parsed_text = None
    value = None

    leading_space = True

    for i in range(pos, len(text)):
        if leading_space and (text[i] in ["-", " "]):
            continue
        leading_space = False
        buf = text[pos : i + 1]
        try:
            value = int(buf)
        except ValueError:
            break
        else:
            parsed_text = buf
    if value is None:
        raise ValueError

    parsed = collections.namedtuple("parsed", "value pos")

    return parsed(value, pos + len(parsed_text))


def get_optimal_objective_value(problem_name):
    if "/" in problem_name:
        path = os.path.splitext(problem_name)[0] + ".txt"
        if os.path.exists(path):
            with open(path, "r") as f:
                return float(f.read())
        else:
            return None

    if "." in problem_name:
        problem_name = problem_name.split(".")[0]
    path = os.path.expanduser("data/SDPLIB/README.md")
    lines = []
    with open(path, "r") as f:
        mode = 0
        for i, l in enumerate(f):
            if mode == 0:
                if l.startswith("|"):
                    mode += 1
            if mode == 1:
                if l.startswith("|"):
                    lines.append(l)
                else:
                    mode += 1
            if mode >= 2:
                break
    lines = lines[:1] + lines[2:]
    lines = [x[2:-3].replace("|", ",") for x in lines]
    lines = "\n".join(lines)
    df = pd.read_csv(io.StringIO(lines), sep=" +, +", engine="python")
    try:
        return float(
            df.loc[
                df["Problem"] == problem_name, "Optimal Objective Value"
            ].iloc[0]
        )
    except:
        return None


if __name__ == "__main__":
    import doctest

    n_failuers, _ = doctest.testmod(
        optionflags=doctest.ELLIPSIS + doctest.NORMALIZE_WHITESPACE
    )
    if n_failuers > 0:
        raise ValueError(f"failed {n_failuers} tests")
