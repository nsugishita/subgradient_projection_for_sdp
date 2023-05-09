# -*- coding: utf-8 -*-

"""Utils on linear algebra"""

import functools

import numpy as np
import scipy.sparse

try:
    import torch

    has_torch = True

except ImportError:
    has_torch = False


def round(x, *args, **kwargs):
    """Round an array to the given number of decimals

    This applies `numpy.round` and replaces all entries close
    to 0 with 0.

    See also
    --------
    numpy.round
    """
    out = np.round(x, *args, **kwargs)
    if np.ndim(out) == 0:
        if np.isclose(out, 0):
            return 0
        else:
            return out
    else:
        out[np.isclose(out, 0)] = 0
        return out


def get_svec_size(n):
    """Given the size of a matrix get the size of svec

    Examples
    --------
    >>> get_svec_size(4)
    10

    Parameters
    ----------
    n : int
        The size of a square matrix. The number of rows (columns).

    Returns
    -------
    size : int
        The size of the output of `svec`.
    """
    return int(n * (n + 1) / 2)


def from_svec_size_to_original_size(n):
    """Given the size of a 'sveced' vector get the original matrix size

    Parameters
    ----------
    size : int
        The size of the output of `svec`.

    Returns
    -------
    n : int
        The size of the original square matrix. The number of rows (columns).
    """
    return int(np.sqrt(2 * n + 1 / 4) - 1 / 2)


@functools.lru_cache(maxsize=None)
def _svec_indices(n, lower_column_major=False):
    """Get indices to compute svec or its inverse

    If lower_column_major is False (default), this extracts the lower
    triangular part row-by-row. If True, this extracts the lower
    triangular part column-by-column.

    >>> A = np.array([
    ...      [  1,  2,  4,  7, 11 ],
    ...      [  2,  3,  5,  8, 12 ],
    ...      [  4,  5,  6,  9, 13 ],
    ...      [  7,  8,  9, 10, 14 ],
    ...      [ 11, 12, 13, 14, 15 ],
    ... ])
    >>> idx = _svec_indices(A.shape[0], lower_column_major=False)
    >>> A[idx['sorter']]
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])

    >>> svec = A[idx['sorter']].astype(float)
    >>> svec[idx['out_idx_diag']] *= np.sqrt(2)
    >>> np.round(svec, 4)
    array([ 1.4142,  2.    ,  4.2426,  4.    ,  5.    ,  8.4853,  7.    ,
            8.    ,  9.    , 14.1421, 11.    , 12.    , 13.    , 14.    ,
           21.2132])

    >>> idx = _svec_indices(A.shape[0], lower_column_major=True)
    >>> A[idx['sorter']]
    array([ 1,  2,  4,  7, 11,  3,  5,  8, 12,  6,  9, 13, 10, 14, 15])

    See also
    --------
    https://jump.dev/MathOptInterface.jl/stable/reference/standard_form/#MathOptInterface.AbstractSymmetricMatrixSetTriangle

    Parameters
    ----------
    n : dimension of the matrix to which svec is applied

    Returns
    -------
    res : dict
        This dict has the following items.
        - in_idx_diag
        - out_idx_diag
        - in_idx_lower
        - out_idx_lower
        - sorter
        - inv_sorter
        We have

        svec(A)[out_idx_diag] == A[in_idx_diag]

        svec(A)[out_idx_lower] == A[in_idx_lower]

        x <- A[sorter]
        x[out_idx_diag] <- coef * x[out_idx_diag]
        svec(A) == x.

        svec(A)[inv_sorter] == A
    """
    out_shape = int(n * (n + 1) / 2)

    if lower_column_major:
        in_idx_diag1 = in_idx_diag2 = np.arange(n)
        out_idx_diag = np.r_[0, np.arange(n, 1, -1)]
        np.cumsum(out_idx_diag, out=out_idx_diag)

        in_idx_lower1 = np.r_[tuple(np.arange(i + 1, n) for i in range(n))]
        in_idx_lower2 = np.r_[tuple(np.full(n - i - 1, i) for i in range(n))]

        out_idx_lower = np.ones(out_shape - n, dtype=int)
        buf = np.arange(n - 1, 1, -1)
        np.cumsum(np.arange(n - 1, 1, -1), out=buf)
        out_idx_lower[buf] = 2
        np.cumsum(out_idx_lower, out=out_idx_lower)

        sorter1 = np.r_[tuple(np.arange(i, n) for i in range(n))]
        sorter2 = np.r_[tuple(np.full(n - i, i) for i in range(n))]

        inv_sorter_buf = np.arange(out_shape)

        inv_sorter = np.empty((n, n), dtype=int)
        inv_sorter[in_idx_diag1, in_idx_diag2] = inv_sorter_buf[out_idx_diag]
        inv_sorter[in_idx_lower1, in_idx_lower2] = inv_sorter_buf[
            out_idx_lower
        ]
        inv_sorter[in_idx_lower2, in_idx_lower1] = inv_sorter_buf[
            out_idx_lower
        ]

    else:
        in_idx_diag1 = in_idx_diag2 = np.arange(n)
        out_idx_diag = np.arange(1, n + 1)
        out_idx_diag[0] = 0
        np.cumsum(out_idx_diag, out=out_idx_diag)

        in_idx_lower1 = np.r_[tuple(np.full(i, i) for i in range(n))]
        in_idx_lower2 = np.r_[tuple(np.arange(i) for i in range(n))]

        out_idx_lower = np.ones(out_shape - n, dtype=int)
        if n > 1:
            buf = np.cumsum(np.arange(n - 1))
            out_idx_lower[buf] = 2
            out_idx_lower[0] = 1
            np.cumsum(out_idx_lower, out=out_idx_lower)

        sorter1 = np.r_[tuple(np.full(i + 1, i) for i in range(n))]
        sorter2 = np.r_[tuple(np.arange(i + 1) for i in range(n))]

        inv_sorter_buf = np.arange(out_shape)

        inv_sorter = np.empty((n, n), dtype=int)
        inv_sorter[in_idx_diag1, in_idx_diag2] = inv_sorter_buf[out_idx_diag]
        inv_sorter[in_idx_lower1, in_idx_lower2] = inv_sorter_buf[
            out_idx_lower
        ]
        inv_sorter[in_idx_lower2, in_idx_lower1] = inv_sorter_buf[
            out_idx_lower
        ]

    return {
        "in_idx_diag": (in_idx_diag1, in_idx_diag2),
        "out_idx_diag": (out_idx_diag,),
        "in_idx_lower": (in_idx_lower1, in_idx_lower2),
        "out_idx_lower": (out_idx_lower,),
        "sorter": (sorter1, sorter2),
        "inv_sorter": (inv_sorter,),
    }


def svec(a, *args, **kwargs):
    """Apply svec operartor on a matrix

    This applies svec operator on a given matrix. svec is defined as

    svec(A) = (A11, c A21, c A31, ..., c An1, A22, c A32, ..., Ann)^T,

    where c = sqrt{2} by default. The operation is broadcasted in
    case of an argument with more than two dimensions.

    Examples
    --------
    One can feed a numpy array or a pytorch tensor.

    >>> import numpy as np
    >>> x = np.array([
    ...     [1., 1., 0., 0.],
    ...     [1., 2., 2., 2.],
    ...     [0., 2., 4., 4.],
    ...     [0., 2., 4., 0.],
    ... ])
    >>> svec(x).round(4)
    array([1.    , 1.4142, 2.    , 0.    , 2.8284, 4.    , 0.    , 2.8284,
           5.6569, 0.    ])

    >>> coo = svec(scipy.sparse.coo_array(x))
    >>> coo
    <10x1 sparse array of type '<class 'numpy.float64'>'
        with 7 stored elements in COOrdinate format>
    >>> coo.toarray().ravel().round(4)
    array([1.    , 1.4142, 2.    , 0.    , 2.8284, 4.    , 0.    , 2.8284,
           5.6569, 0.    ])

    Parameters
    ----------
    a : array of 2 or more dimensions
    coef : float, default np.sqrt(2)
        The coefficient multiplied to the non-diagonal entries.
        Set None to prevent any scaling.
    out : array, optional
        If given, the output is written on this array.
        This is only supported when `a` is a numpy array.

    Returns
    -------
    result : array
    """
    if has_torch and isinstance(a, torch.Tensor):
        return svec_torch(a, *args, **kwargs)
    elif isinstance(a, scipy.sparse.spmatrix):
        return svec_scipy_sparse(a)
    else:
        return svec_numpy(a, *args, **kwargs)


def svec_numpy(a, coef=np.sqrt(2), out=None):
    """Apply svec operator on a numpy array

    For detail see `svec`.
    """
    n = a.shape[-1]
    if out is None:
        out_shape = a.shape[:-2] + (int(n * (n + 1) / 2),)
        out = np.empty(out_shape)
    indices = _svec_indices(n)
    out_idx_diag = indices["out_idx_diag"]
    in_idx_diag = indices["in_idx_diag"]
    out[(Ellipsis,) + out_idx_diag] = a[(Ellipsis,) + in_idx_diag]
    out_idx_lower = indices["out_idx_lower"]
    in_idx_lower = indices["in_idx_lower"]
    if (coef is None) or (coef == 1):
        out[(Ellipsis,) + out_idx_lower] = a[(Ellipsis,) + in_idx_lower]
    else:
        out[(Ellipsis,) + out_idx_lower] = coef * a[(Ellipsis,) + in_idx_lower]
    return out


def svec_torch(a, coef=np.sqrt(2)):
    """Apply svec operator on a pytorch tensor

    For detail see `svec`.
    """
    n = a.shape[-1]
    indices = _svec_indices(n)
    a = 0.5 * (
        a[(Ellipsis,) + indices["sorter"]]
        + a[(Ellipsis,) + indices["sorter"][::-1]]
    )
    _coef = torch.ones(get_svec_size(n))
    _coef[indices["out_idx_lower"]] = coef
    a = _coef * a
    return a


def svec_scipy_sparse(*args, lower_column_major=False, **kwargs):
    """Compute svec from lower triangular and diagonal entries

    This creates svec from a square matrix. Only entries on
    lower triangular and diagonal is used.

    Examples
    --------
    >>> import scipy.sparse
    >>> row = [1, 1, 2, 3, 3]
    >>> col = [0, 1, 1, 1, 3]
    >>> data = [-1, 1, 2, 1, -1]
    >>> a = scipy.sparse.coo_matrix((data, (row, col)), shape=(4, 4))
    >>> print(a.todense())
    [[ 0  0  0  0]
     [-1  1  0  0]
     [ 0  2  0  0]
     [ 0  1  0 -1]]
    >>> b = svec_scipy_sparse(a)
    >>> print(b.todense().T.round(2))
    [[ 0.   -1.41  1.    0.    2.83  0.    0.    1.41  0.   -1.  ]]
    >>> b = svec_scipy_sparse(a, lower_column_major=True)
    >>> print(b.todense().T.round(2))
    [[ 0.   -1.41  0.    0.    1.    2.83  1.41  0.    0.   -1.  ]]

    Entries in the upper triangular part are ignored.

    >>> row = [1, 1, 2]
    >>> col = [0, 1, 3]
    >>> data = [1, 2, 3]
    >>> a = scipy.sparse.coo_matrix((data, (row, col)), shape=(4, 4))
    >>> print(a.todense())
    [[0 0 0 0]
     [1 2 0 0]
     [0 0 0 3]
     [0 0 0 0]]
    >>> b = svec_scipy_sparse(a)
    >>> print(b.todense().T.round(2))
    [[0.   1.41 2.   0.   0.   0.   0.   0.   0.   0.  ]]
    >>> b = svec_scipy_sparse(a, lower_column_major=True)
    >>> print(b.todense().T.round(2))
    [[0.   1.41 0.   0.   2.   0.   0.   0.   0.   0.  ]]
    """
    if lower_column_major:
        return _svec_scipy_sparse_column_major(*args, **kwargs)
    else:
        return _svec_scipy_sparse_row_major(*args, **kwargs)


def _svec_scipy_sparse_row_major(a, format=None, coef=None):
    if coef is None:
        coef = np.sqrt(2)

    original_type = a.__class__
    a = a.tocoo()

    if format is not None:
        raise NotImplementedError("format != None is not yet supported")
    if isinstance(a, scipy.sparse.coo_matrix):
        n = a.shape[0]
        row = a.row
        col = a.col
        data = a.data
        # Ignore entries on the upper triangular part.
        selector = np.nonzero(row >= col)[0]
        if selector.size < row.size:
            row = row[selector]
            col = col[selector]
            data = data[selector]
        # Compute the row and column indices of entries in svec.
        new_col = np.full(col.size, 0, dtype=np.int32)
        new_row = row * (row + 1) / 2 + col
        new_n_rows = int(n * (n + 1) / 2)
        if coef == 1:
            new_data = data
        else:
            new_data = data.copy()
            if "int" in str(new_data.dtype):
                new_data = new_data.astype(float)
            new_data[row != col] *= coef
        res = scipy.sparse.coo_matrix(
            (new_data, (new_row, new_col)), shape=(new_n_rows, 1)
        )
        return original_type(res, shape=res.shape)

    else:
        raise NotImplementedError(f"unsupported type: {type(a)}")


def _svec_scipy_sparse_column_major(a, format=None, coef=None):
    if coef is None:
        coef = np.sqrt(2)

    original_type = a.__class__
    a = a.tocoo()

    if format is not None:
        raise NotImplementedError("format != None is not yet supported")
    if isinstance(a, scipy.sparse.coo_matrix):
        n = a.shape[0]
        row = a.row
        col = a.col
        data = a.data
        # Ignore entries on the upper triangular part.
        selector = np.nonzero(row >= col)[0]
        if selector.size < row.size:
            row = row[selector]
            col = col[selector]
            data = data[selector]
        # Compute the row and column indices of entries in svec.
        new_col = np.full(col.size, 0, dtype=np.int32)
        distance_from_diagonal = row - col
        offset_by_column = np.r_[0, np.cumsum(np.arange(n, 0, -1))]
        new_row = distance_from_diagonal + offset_by_column[col]
        new_n_rows = int(n * (n + 1) / 2)
        if coef == 1:
            new_data = data
        else:
            new_data = data.copy()
            if "int" in str(new_data.dtype):
                new_data = new_data.astype(float)
            new_data[row != col] *= coef
        res = scipy.sparse.coo_matrix(
            (new_data, (new_row, new_col)), shape=(new_n_rows, 1)
        )
        return original_type(res, shape=res.shape)

    else:
        raise NotImplementedError(f"unsupported type: {type(a)}")


def svec_inv(a, *args, **kwargs):
    """Apply the inverse of svec operartor on a vector

    This applies the inverse of svec operator on a given verctor. In other
    words, this reconstructs the original matrix given the result of svec
    operator. svec is defined as

    svec(A) = (A11, c A21, c A31, ..., c An1, A22, c A32, ..., Ann)^T,

    where c = sqrt{2} by default. The operation is broadcasted in
    case of an argument with more than two dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([
    ...     [1., 1., 0., 0.],
    ...     [1., 2., 2., 2.],
    ...     [0., 2., 4., 4.],
    ...     [0., 2., 4., 0.],
    ... ])
    >>> s = svec(x)
    >>> svec_inv(s)
    array([[1., 0., 0., 0.],
           [1., 2., 0., 0.],
           [0., 2., 4., 0.],
           [0., 2., 4., 0.]])

    Parameters
    ----------
    a : array
    coef : float, default np.sqrt(2)
        The coefficient used in svec operator
    part : {'l', 'u', 'f'}
        This indicates which part of the matrix should be written.
        'l' is lower and diagonal, 'u' is upper and diagonal and 'f'
        is full.
    out : array, optional
        If given, the output is written on this array.

    Returns
    -------
    result : array
    """
    if has_torch and isinstance(a, torch.Tensor):
        return svec_inv_torch(a, *args, **kwargs)
    else:
        return svec_inv_numpy(a, *args, **kwargs)


def svec_inv_numpy(a, coef=np.sqrt(2), part="l", out=None):
    """Apply the inverse of svec operartor on a numpy array

    See `svec_inv` for doc.
    """
    if part not in "luf":
        raise ValueError(f"{part=}")
    # Get the dimension of the original matrix.
    n = from_svec_size_to_original_size(a.shape[-1])
    if out is None:
        out_shape = a.shape[:-1] + (n, n)
        out = np.zeros(out_shape)
    indices = _svec_indices(n)
    out_idx_diag = indices["out_idx_diag"]
    in_idx_diag = indices["in_idx_diag"]
    out[(Ellipsis,) + in_idx_diag] = a[(Ellipsis,) + out_idx_diag]
    out_idx_lower = indices["out_idx_lower"]
    in_idx_lower = indices["in_idx_lower"]
    if part in ["l", "f"]:
        if (coef is None) or (coef == 1):
            out[(Ellipsis,) + in_idx_lower] = a[(Ellipsis,) + out_idx_lower]
        else:
            cinv = 1 / coef
            out[(Ellipsis,) + in_idx_lower] = (
                cinv * a[(Ellipsis,) + out_idx_lower]
            )
    if part in ["u", "f"]:
        if (coef is None) or (coef == 1):
            out[(Ellipsis,) + in_idx_lower[::-1]] = a[
                (Ellipsis,) + out_idx_lower
            ]
        else:
            cinv = 1 / coef
            out[(Ellipsis,) + in_idx_lower[::-1]] = (
                cinv * a[(Ellipsis,) + out_idx_lower]
            )
    return out


def svec_inv_torch(a, coef=np.sqrt(2)):
    """Apply the inverse of svec operartor on a numpy array

    See `svec_inv` for doc.
    """
    n = from_svec_size_to_original_size(a.shape[-1])
    indices = _svec_indices(n)
    out = a[(Ellipsis,) + indices["inv_sorter"]]
    coef = torch.full(out.shape[-2:], 1 / coef)
    coef[np.arange(n), np.arange(n)] = 1
    out = out * coef
    return out


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)

# vimquickrun: . ./scripts/activate.sh && python %
