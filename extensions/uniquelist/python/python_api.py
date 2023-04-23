# -*- coding: utf-8 -*-

"""Interface to uniquelist."""

import ctypes

import numpy as np


class LibManager(object):
    """Stub to use uniquelist"""

    __slots__ = ("lib",)

    def __init__(self):
        """Initialise an LibManager instance"""
        self.lib = ctypes.CDLL("libuniquelist.so")
        _set_signature(self.lib)

    def is_available(self):
        """Test if uniquelist is available"""
        return True

    def create_unique_array_list(self):
        return unique_array_list(self.lib)

    def create_batched_unique_array_list(self, batch_size, array_size):
        return batched_unique_array_list(self.lib, batch_size, array_size)


class unique_array_list(object):
    def __init__(self, lib):
        """Initialise an unique_array_list instance

        Parameters
        ----------
        lib
        """
        self.lib = lib
        self.batch_size = 1
        self.list = self.lib.uniquelist_batched_unique_array_list_create(
            self.batch_size
        )

    def __del__(self):
        self.lib.uniquelist_batched_unique_array_list_delete(
            self.list, self.batch_size
        )

    @property
    def size(self):
        res = np.empty(self.batch_size, dtype=np.int32)
        self.lib.uniquelist_batched_unique_array_list_size(
            self.list, self.batch_size, res
        )
        return res[0]

    def append(self, value):
        """Append an array to the end

        This appends an array to the end.  If the key is already
        in the list, this does not modify anything.

        Parameters
        ----------
        value : (n_items, array_size) array of float

        Returns
        -------
        pos : (n_items,) array of int
            Position of the arrays in the lists.
        new : (n_items,) array of bool
            True if given arrays are new and False otherwise.
        """
        value = np.atleast_2d(np.asarray(value, dtype=float))
        list_index = np.repeat(0, value.shape[0])
        pos = np.empty((list_index.size,), dtype=np.int32)
        new = np.empty((list_index.size,), dtype=np.int32)
        self.lib.uniquelist_batched_unique_array_list_push_back(
            self.list,
            value.shape[1],
            list_index.size,
            list_index,
            value.ravel(),
            pos,
            new,
        )
        return pos, new.astype(bool)

    def erase_nonzero(self, flag):
        """Remove items at the positions of nonzero elements

        Parameters
        ----------
        flag : array of bool or int
        """
        flag = np.asarray(flag, dtype=np.int32)
        list_index = np.repeat(0, flag.size)
        self.lib.uniquelist_batched_unique_array_list_erase_nonzero(
            self.list,
            list_index,
            flag.size,
            flag,
        )

    def isin(self, value):
        """Test whether given arrays are in the list or not

        Parameters
        ----------
        value : (n_items, array_size) array of float

        Returns
        -------
        result : (n_items,) array of bool
            True if given arrays are found in the list
        """
        value = np.atleast_2d(np.asarray(value, dtype=float))
        list_index = np.repeat(0, value.shape[0])
        result = np.empty((list_index.size,), dtype=bool)
        self.lib.uniquelist_batched_unique_array_list_isin(
            self.list,
            value.shape[1],
            list_index.size,
            list_index,
            value.ravel(),
            result,
        )
        return result

    def dump(self):
        """Print the current status of the list on the console"""
        self.lib.uniquelist_batched_unique_array_list_dump(self.list, 0)


class batched_unique_array_list(object):
    def __init__(self, lib, batch_size, array_size):
        """Initialise an batched_unique_array_list instance

        Parameters
        ----------
        lib
        batch_size : int
            Number of lists.
        """
        self.lib = lib
        self.batch_size = batch_size
        self.array_size = array_size
        self.list = self.lib.uniquelist_batched_unique_array_list_create(
            batch_size, array_size
        )

    def __del__(self):
        self.lib.uniquelist_batched_unique_array_list_delete(
            self.list, self.batch_size
        )

    @property
    def size(self):
        res = np.empty(self.batch_size, dtype=np.int32)
        self.lib.uniquelist_batched_unique_array_list_size(
            self.list, self.batch_size, res
        )
        return res

    def append(self, list_index, value):
        """Append an array to the end

        This appends an array to the end.  If the key is already
        in the list, this does not modify anything.

        Parameters
        ----------
        list_index : 1d array of int or int
        value : (list_index.size, array_size) array of float

        Returns
        -------
        pos : (list_index.size,) array of int
            Position of the arrays in the lists.
        new : (list_index.size,) array of bool
            True if given arrays are new and False otherwise.
        """
        list_index = np.atleast_1d(np.asarray(list_index, dtype=np.int32))
        value = np.asarray(value, dtype=float)
        squeeze_result = value.ndim <= 1
        value = np.atleast_2d(value)
        if list_index.shape[0] == 1:
            list_index = np.repeat(list_index, value.shape[0])
        pos = np.empty((list_index.size,), dtype=np.int32)
        new = np.empty((list_index.size,), dtype=np.int32)
        self.lib.uniquelist_batched_unique_array_list_push_back(
            self.list,
            value.shape[1],
            list_index.size,
            list_index,
            value.ravel(),
            pos,
            new,
        )
        new = new.astype(bool)
        if squeeze_result:
            pos = pos[0]
            new = new[0]
        return pos, new

    def erase_nonzero(self, list_index, flag):
        """Remove items at the positions of nonzero elements

        Parameters
        ----------
        list_index : int
        flag : array of bool or int
        """
        flag = np.asarray(flag, dtype=np.int32)
        self.lib.uniquelist_batched_unique_array_list_erase_nonzero(
            self.list,
            list_index,
            flag.size,
            flag,
        )

    def isin(self, list_index, value):
        """Test whether given arrays are in the list or not

        Parameters
        ----------
        list_index : 1d array of int or int
        value : (list_index.size, array_size) array of float

        Returns
        -------
        result : (list_index.size,) array of bool
            True if given arrays are found in the list
        """
        list_index = np.atleast_1d(np.asarray(list_index, dtype=np.int32))
        value = np.atleast_2d(np.asarray(value, dtype=float))
        if list_index.shape[0] == 1:
            list_index = np.repeat(list_index, value.shape[0])
        result = np.empty((list_index.size,), dtype=bool)
        self.lib.uniquelist_batched_unique_array_list_isin(
            self.list,
            value.shape[1],
            list_index.size,
            list_index,
            value.ravel(),
            result,
        )
        return result

    def dump(self, list_index):
        """Print the current status of the list on the console"""
        self.lib.uniquelist_batched_unique_array_list_dump(
            self.list, list_index
        )


def _set_signature(lib):
    lib.uniquelist_batched_unique_array_list_create.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.uniquelist_batched_unique_array_list_create.restype = ctypes.c_void_p

    lib.uniquelist_batched_unique_array_list_delete.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
    ]
    lib.uniquelist_batched_unique_array_list_delete.restype = ctypes.c_void_p

    lib.uniquelist_batched_unique_array_list_size.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        np.ctypeslib.ndpointer(dtype=np.int32),
    ]
    lib.uniquelist_batched_unique_array_list_size.restype = ctypes.c_void_p

    lib.uniquelist_batched_unique_array_list_push_back.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,  # array_size
        ctypes.c_int32,  # n_items
        np.ctypeslib.ndpointer(dtype=np.int32),  # list_index
        np.ctypeslib.ndpointer(dtype=float),  # value
        np.ctypeslib.ndpointer(dtype=np.int32),  # pos
        np.ctypeslib.ndpointer(dtype=np.int32),  # new
    ]
    lib.uniquelist_batched_unique_array_list_push_back.restype = (
        ctypes.c_void_p
    )

    lib.uniquelist_batched_unique_array_list_erase_nonzero.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        np.ctypeslib.ndpointer(dtype=np.int32),
    ]
    lib.uniquelist_batched_unique_array_list_erase_nonzero.restype = (
        ctypes.c_void_p
    )

    lib.uniquelist_batched_unique_array_list_isin.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,  # array_size
        ctypes.c_int32,  # n_items
        np.ctypeslib.ndpointer(dtype=np.int32),  # list_index
        np.ctypeslib.ndpointer(dtype=float),  # value
        np.ctypeslib.ndpointer(dtype=bool),  # result
    ]
    lib.uniquelist_batched_unique_array_list_isin.restype = ctypes.c_void_p

    lib.uniquelist_batched_unique_array_list_dump.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
    ]
    lib.uniquelist_batched_unique_array_list_dump.restype = ctypes.c_void_p


uniquelist = LibManager()
