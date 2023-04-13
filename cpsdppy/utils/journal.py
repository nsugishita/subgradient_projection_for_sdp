# -*- coding: utf-8 -*-

"""Utilities."""

import copy
import time
import typing

import numpy as np

_missing: dict = {}


class IterationJournal:
    """Logger to save statistics bound to iterations

    This is a logger to save information related to iterations,
    such as step size.  This can save not only scalars but also
    arrays.  This keeps data as lists internally and saves values
    with the time they get available.

    Items stored in this instance must be registered with
    `register_iteration_items` method first.
    """

    def __init__(self, config=None) -> None:
        """Initialise an IterationJournal instance"""
        self.iteration_item_values: typing.Dict[str, typing.Any] = dict()
        self.iteration_item_times: typing.Dict[str, typing.Any] = dict()
        self.iteration_item_default_values: typing.Dict[
            str, typing.Any
        ] = dict()
        self.iteration_item_with_timing: typing.List[str] = []
        self.start_hook()

    def start_hook(self):
        """Notify the start of the solver"""
        self.starttime = time.perf_counter()
        self.iteration = 0

    def iteration_start_hook(self, iteration):
        """Notify the start of a new iteration"""
        self.iteration = iteration
        to = self.iteration + 1
        for key in self.iteration_item_default_values:
            self._append_item(key, to=to)

    def _append_item(self, key, value=_missing, time=np.nan, n=None, to=None):
        """Extend the list by appending items

        Parameters
        ----------
        key : str
        value : object, optional
            If omitted, the default value is used.
        time : float, default np.nan
        n : int, optional
            If given, this specifies the number of items to be appended.
            If `n` and `to` is omitted, only one item is appended.
        to : int, optional
            If given, this specifies the expected length of the list
            ater appending items.  If `n` is given, this is ignored.
        """
        if value is _missing:
            value = self.iteration_item_default_values[key]
        self.iteration_item_values.setdefault(key, [])
        if key in self.iteration_item_with_timing:
            self.iteration_item_times.setdefault(key, [])
        if n is None:
            if to is None:
                n = 1
            else:
                n = to - len(self.iteration_item_values[key])
        if key in self.iteration_item_with_timing:
            for i in range(n):
                self.iteration_item_values[key].append(copy.deepcopy(value))
                self.iteration_item_times[key].append(time)
        else:
            for i in range(n):
                self.iteration_item_values[key].append(copy.deepcopy(value))

    def register_iteration_items(self, **kwargs):
        """Register items to be logged

        This registers items to be logged.  This method should
        be called with the following signature:

        ```
        register_iteration_items(
            item_name1=default_value1,
            item_name2=default_value2,
            ...
        )
        ```

        To control the behaviour finely, one can pass a dict.
        It may contain the following keys:
        - default (required) : object
            Default value
        - timing : bool, default True
            Keep timing or not.
        """
        for key, value in kwargs.items():
            if not isinstance(value, dict):
                value = dict(default=value)
            assert "default" in value
            self.iteration_item_default_values[key] = value["default"]
            if value.get("timing", True):
                self.iteration_item_with_timing.append(key)

    def set_iteration_items(self, **kwargs):
        """Set values of items

        This saves values of given items.  This method should
        be called with the following signature:

        ```
        set_iteration_items(
            item_name1=saved_value1,
            item_name2=saved_value2,
            ...
        )
        ```
        """
        it = self.iteration
        to = it + 1
        elapse = time.perf_counter() - self.starttime
        for key, value in kwargs.items():
            self._append_item(key, to=to)
            self.iteration_item_values[key][self.iteration] = value
            if key in self.iteration_item_with_timing:
                self.iteration_item_times[key][self.iteration] = elapse

    def is_iteration_item(self, key):
        return key in self.iteration_item_default_values

    def get(
        self,
        key,
        default=_missing,
        iteration=_missing,
        return_default_on_index_error=False,
    ):
        if not self.is_iteration_item(key):
            if default is _missing:
                raise KeyError(key)
            else:
                return default
        self._append_item(key, to=self.iteration + 1)
        if iteration is _missing:
            iteration = -1
        try:
            return self.iteration_item_values[key][iteration]
        except IndexError as e:
            if return_default_on_index_error:
                return self.iteration_item_default_values[key]
            else:
                raise e from None

    def get_all(self, key):
        return self.iteration_item_values[key]

    def add_iteration_item(self, key, value):
        self._append_item(key, to=self.iteration + 1)
        self.iteration_item_values[key][-1] += value

    def dump_data(
        self,
        out=None,
        value_format="iter_{key}",
        time_format="iter_{key}_time",
    ):
        """Output all saved data

        Parameters
        ----------
        out : dict, optional
            If given, the results are written on this dict.
            Otherwise, a new dict is returned.
        value_format : str, default "iter_{key}"
            Format string to define an item name in the output.
        time_format : str, default "iter_{key}_time"
            Format string to define an item name in the output.

        Returns
        -------
        res : dict, optional
            This is returned when `out` is not given.
        """
        if out is None:
            return_result = True
            out = {}
        else:
            return_result = False
        for key, value in self.iteration_item_values.items():
            if isinstance(self.iteration_item_default_values[key], list):
                out[value_format.format(key=key)] = np.concatenate(value)
            else:
                out[value_format.format(key=key)] = value
            if key in self.iteration_item_with_timing:
                out[time_format.format(key=key)] = self.iteration_item_times[
                    key
                ]
        if return_result:
            return out


if __name__ == "__main__":
    import doctest

    doctest.testmod()
