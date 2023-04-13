# -*- coding: utf-8 -*-

"""Utility to measure time"""

import typing


class Missing(object):
    """Object to represent missing arguments"""

    def __repr__(self):
        return "missing"

    def __str__(self):
        return "missing"


missing = Missing()


class timer:
    """Utility to measure time

    This is a context manager to facilitate time measurement.

    Examples
    --------
    In the examples below we use `test_time_stub`, which is
    only required for testing purposes. In the actual use
    this is not needed.

    >>> set_test_time_stub()

    Timer can be used to measure how long the program takes
    using the 'with' block.

    >>> with timer() as t:
    ...    # Pretend time has advanced.
    ...    test_time_stub.advance(1)
    >>> t.walltime  # get the length in wall time.
    1.0
    >>> t.proctime  # get the length in process time.
    1.0

    Timer can be used to measure multiple routines as well.
    The duration and start time of each routine is available in
    `records['walltime']` and `records['start_walltime']`.

    >>> t = timer()  # start measuring.
    >>> for i in range(3):
    ...     t.iteration = i
    ...     with t.routine('foo'):
    ...         test_time_stub.advance(1)
    ...     with t.routine('bar'):
    ...         test_time_stub.advance(0.5)
    >>> t.stop()  # manually stop the timer
    >>> t.records['type']
    ['foo', 'bar', 'foo', 'bar', 'foo', 'bar']
    >>> t.records['iteration']
    [0, 0, 1, 1, 2, 2]
    >>> t.records['start_walltime']
    [0.0, 1.0, 1.5, 2.5, 3.0, 4.0]
    >>> t.records['end_walltime']
    [1.0, 1.5, 2.5, 3.0, 4.0, 4.5]
    >>> t.records['walltime']
    [1.0, 0.5, 1.0, 0.5, 1.0, 0.5]

    One can nest the measurement (nest the `with` block). The nest
    level of the measurement is available in `records['nest_level']`.

    >>> t = timer()  # start measuring.
    >>> for i in range(2):
    ...     t.iteration = i
    ...     with t.routine('foo'):
    ...         with t.routine('bar'):
    ...             pass
    ...         with t.routine('baz'):
    ...             pass
    >>> t.stop()  # manually stop the timer
    >>> t.records['type']
    ['foo', 'bar', 'baz', 'foo', 'bar', 'baz']
    >>> t.records['iteration']
    [0, 0, 0, 1, 1, 1]
    >>> t.records['nest_level']
    [0, 1, 1, 0, 1, 1]

    One can set additional items using `timer.set(key, value)`.
    The values will be saved in `reords[key]`.

    >>> with timer() as t:
    ...     t.set("checkpoint", 0)
    ...     for i in range(4):
    ...         t.iteration = i
    ...         with t.routine("foo"):
    ...             pass  # do somehing
    ...         if i >= 2:
    ...             t.set("checkpoint", 1)
    ...         with t.routine("bar"):
    ...             pass  # do another thing
    >>> t.records["iteration"]
    [0, 0, 1, 1, 2, 2, 3, 3]
    >>> t.records["checkpoint"]
    [0, 0, 0, 0, 0, 1, 1, 1]

    If `non_overwrapping_routines=True`, calling `routine` stops
    a previous routine. It is convenient to avoid using contexts.

    >>> t = timer()
    >>> t.non_overwrapping_routines = True
    >>> _ = t.routine('foo')
    >>> test_time_stub.advance(1)
    >>> _ = t.routine('bar')
    >>> test_time_stub.advance(1)
    >>> t.stop()
    >>> t.records['type']
    ['foo', 'bar']
    >>> t.records['walltime']
    [1.0, 1.0]

    If `non_overwrapping_routines` is False, all the routines
    are stopped when `stop` is called.

    >>> t = timer()
    >>> _ = t.routine('foo')
    >>> test_time_stub.advance(1)
    >>> _ = t.routine('bar')
    >>> test_time_stub.advance(1)
    >>> t.stop()
    >>> t.records['type']
    ['foo', 'bar']
    >>> t.records['walltime']
    [2.0, 1.0]
    """

    __slots__ = (
        "time_logic",
        "start_walltime",
        "start_proctime",
        "end_walltime",
        "end_proctime",
        "current_data",
        "ongoing_record_index",
        "records",
        "non_overwrapping_routines",
    )

    def __init__(self) -> None:
        """Initialise a time instance"""
        self.time_logic = time_routine
        walltime, proctime = self.time_logic()
        self.start_walltime: float = walltime
        self.start_proctime: float = proctime
        self.end_walltime: typing.Optional[float] = None
        self.end_proctime: typing.Optional[float] = None
        self.non_overwrapping_routines = False

        self.records: typing.Dict = dict(
            iteration=[],
            type=[],
            nest_level=[],
            start_walltime=[],
            start_proctime=[],
            end_walltime=[],
            end_proctime=[],
            walltime=[],
            proctime=[],
        )
        self.current_data = dict(
            iteration=0,
            type=None,
            start_walltime=-1,
            start_proctime=-1,
        )
        self.ongoing_record_index: typing.List = list()

    @property
    def walltime(self) -> float:
        """Get duration in wall time"""
        if self.end_walltime is not None:
            return self.end_walltime - self.start_walltime
        else:
            return self.time_logic()[0] - self.start_walltime

    @property
    def proctime(self) -> float:
        """Get duration in process time"""
        if self.end_proctime is not None:
            return self.end_proctime - self.start_proctime
        else:
            return self.time_logic()[1] - self.start_proctime

    @property
    def iteration(self) -> int:
        """Get iteration index"""
        return typing.cast(int, self.current_data["iteration"])

    @iteration.setter
    def iteration(self, iteration) -> None:
        """Set iteration index"""
        self.current_data["iteration"] = iteration

    def iterations(self, iterations):
        """Set iteration index as the loop progress"""
        for i in iterations:
            self.iteration = i
            yield i

    def set(self, key, value, default=missing, update_onging_records=False):
        """Set additional item to save in records

        This adds additional item to the timer.

        Examples
        --------
        Adding new items after calling `self.routine` results
        in a ValueError.

        >>> t = timer()
        >>> with t.routine("initialisation"):
        ...    pass
        >>> t.set("checkpoint", 0)
        Traceback (most recent call last):
        ...
        ValueError: adding new item 'checkpoint' after measurements are started

        One can set a default value in such a case.

        >>> t = timer()
        >>> with t.routine("initialisation"):
        ...    pass
        >>> t.set("checkpoint", 0, default=-1)
        >>> with t.routine("main routine"):
        ...    pass
        >>> t.stop()
        >>> t.records["checkpoint"]
        [-1, 0]

        `update_onging_records` contols  the behaviour when `set`
        is called while some records are being measured.

        >>> t = timer()
        >>> t.set("checkpoint", 0)
        >>> with t.routine("initialisation"):
        ...    pass
        >>> with t.routine("main routine"):
        ...    t.set("checkpoint", 1)
        >>> with t.routine("post processing"):
        ...    t.set("checkpoint", 2, update_onging_records=True)
        >>> t.stop()
        >>> t.records["checkpoint"]
        [0, 0, 2]

        Parameters
        ----------
        key : str
            Name of the item.
        value : obj
            Value of the item.
        default : obj, optional
            Default values used for records already created.
        update_onging_records : bool, default False
            When this is True, the values of ongoing records are
            updated to the given value.
        """
        if key not in self.records:
            # New item is being added.
            if len(self.records["type"]) > 0:
                # Measurements already started.
                if default is missing:
                    raise ValueError(
                        f"adding new item '{key}' after measurements "
                        "are started"
                    )
                else:
                    # Set the default value on all the records
                    # created already.
                    self.records[key] = [default] * len(self.records["type"])
            else:
                # Measurements are not yet started.
                self.records[key] = []
        if update_onging_records:
            # For those currently being measured, set the given value.
            for i in self.ongoing_record_index:
                self.records[key][i] = value
        self.current_data[key] = value

    def __enter__(self, *args, **kwargs):  # type: (object, object) -> timer
        """Do nothing but return itself and enter the context"""
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """Record exit time of the context"""
        self.stop()

    def reset_starttime(self):
        """Set the current time as the beginning"""
        walltime, proctime = self.time_logic()
        self.start_walltime = walltime
        self.start_proctime = proctime

    def stop(self):
        """Finish the whole measurement"""
        self.stop_all()
        if self.end_walltime is None:
            walltime, proctime = self.time_logic()
            self.end_walltime = walltime
            self.end_proctime = proctime

    def routine(self, type, **kwargs):
        """Start measuring a routine"""
        if self.non_overwrapping_routines:
            self.stop_routine(not_found_ok=True)
        snapshot = {**self.current_data, **kwargs}
        walltime, proctime = self.time_logic()
        snapshot["start_walltime"] = walltime - self.start_walltime
        snapshot["start_proctime"] = proctime - self.start_proctime
        snapshot["end_walltime"] = float("nan")
        snapshot["end_proctime"] = float("nan")
        snapshot["walltime"] = float("nan")
        snapshot["proctime"] = float("nan")
        snapshot["type"] = type
        snapshot["nest_level"] = len(self.ongoing_record_index)
        for key, value in snapshot.items():
            self.records[key].append(value)
        record_index = len(self.records["type"]) - 1
        context = TimerRoutineMeasureContext(self, record_index)
        self.ongoing_record_index.append(record_index)
        return context

    def stop_routine(self, not_found_ok=False):
        """Stop measuring the routine added most recently"""
        if len(self.ongoing_record_index) > 0:
            record_index = self.ongoing_record_index[-1]
        else:
            record_index = -1
        self.stop_routine_by_record_index(
            record_index, not_found_ok=not_found_ok
        )

    def stop_routine_by_record_index(self, record_index, not_found_ok=False):
        measured = record_index in self.ongoing_record_index
        if not measured:
            if not not_found_ok:
                raise ValueError(
                    f"given record index '{record_index}' is not being measure"
                ) from None
            else:
                return
        self.ongoing_record_index.remove(record_index)
        walltime, proctime = self.time_logic()
        self.records["end_walltime"][record_index] = (
            walltime - self.start_walltime
        )
        self.records["end_proctime"][record_index] = (
            proctime - self.start_proctime
        )
        self.records["walltime"][record_index] = (
            self.records["end_walltime"][record_index]
            - self.records["start_walltime"][record_index]
        )
        self.records["proctime"][record_index] = (
            self.records["end_proctime"][record_index]
            - self.records["start_proctime"][record_index]
        )

    def stop_all(self):
        """Stop all the measurements"""
        while True:
            try:
                self.stop_routine()
            except ValueError:
                break

    def dump_data(self, out=None, prefix="timerecord_", use_numpy=True):
        """Output data

        This outputs data on a given dict or return as a new dict.

        Examples
        --------
        In the examples below we use `test_time_stub`, which is
        only required for testing purposes. In the actual use
        this is not needed.

        >>> set_test_time_stub()
        >>> t = timer()  # start measuring.
        >>> with t.routine('foo'):
        ...     test_time_stub.advance(1)
        >>> with t.routine('bar'):
        ...     test_time_stub.advance(0.5)
        >>> t.stop()  # manually stop the timer
        >>> import pprint
        >>> pprint.pp(t.dump_data())
        {'walltime': 1.5,
         'proctime': 1.5,
         'start_walltime': 9.5,
         'start_proctime': 9.5,
         'end_walltime': 11.0,
         'end_proctime': 11.0,
         'timerecord_iteration': array([0, 0]),
         'timerecord_type': array(['foo', 'bar'], dtype='<U3'),
         'timerecord_nest_level': array([0, 0]),
         'timerecord_start_walltime': array([0., 1.]),
         'timerecord_start_proctime': array([0., 1.]),
         'timerecord_end_walltime': array([1. , 1.5]),
         'timerecord_end_proctime': array([1. , 1.5]),
         'timerecord_walltime': array([1. , 0.5]),
         'timerecord_proctime': array([1. , 0.5]),
         'n_timerecords': 2}

        Parameters
        ----------
        out : dict, optional
            If given, the results are written on this dict.
            Otherwise, a new dict is returned.
        prefix : str, defaulot 'timerecord_'
            Prefix added to the item names.
        use_numpy : bool, default True
            Use numpy array in the output.

        Returns
        -------
        res : dict, optional
            This is returned when `out` is not given.
        """
        if use_numpy:
            import numpy as np

        if out is None:
            return_result = True
            out = {}
        else:
            return_result = False

        out["walltime"] = self.walltime
        out["proctime"] = self.proctime
        out["start_walltime"] = self.start_walltime
        out["start_proctime"] = self.start_proctime
        out["end_walltime"] = self.end_walltime
        out["end_proctime"] = self.end_proctime
        value = []
        prefix = str(prefix)
        if use_numpy:
            cast = np.array
        else:

            def identity(x):
                return x

            cast = identity
        for key, value in self.records.items():
            out[prefix + key] = cast(value)
        out["n_timerecords"] = len(value)

        if return_result:
            return out


class TimerRoutineMeasureContext(object):
    """Context to manage a timer"""

    def __init__(self, timer, record_index) -> None:
        """Initialise an TimerRoutineMeasureContext instance."""
        self.timer = timer
        self.record_index = record_index

    def __enter__(self, *args, **kwargs) -> timer:
        """Enter the context

        Returns
        -------
        timer
        """
        return self.timer

    def __exit__(self, *args, **kwargs) -> None:
        """Stop the timer"""
        self.timer.stop_routine_by_record_index(self.record_index)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.record_index})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.record_index})"


def _actual_time_routine():
    """Return walltime and proctime

    Returns
    -------
    walltime : float
    proctime : float
    """
    import time

    return time.perf_counter(), time.process_time()


class TestTimeStab(object):
    """A stub to return 'time' for a testing purpose"""

    def __init__(self):
        """Initialise a test_time_stub instance"""
        self.t = 0.0

    def set(self, t):
        """Set the current 'time'"""
        self.t = t

    def advance(self, t):
        """Advance the current 'time'"""
        self.t += t

    def __call__(self):
        """Return 'walltime' and 'proctime'

        Returns
        -------
        walltime : float
        proctime : float
        """
        return self.t, self.t


test_time_stub = TestTimeStab()


def set_test_time_stub():
    """Use test_time_stub to measure time

    See also
    --------
    unset_test_time_stub
    """
    global time_routine

    time_routine = test_time_stub


def unset_test_time_stub():
    """Use 'time' module to measure time

    See also
    --------
    set_test_time_stub
    """
    global time_routine

    time_routine = _actual_time_routine


time_routine = _actual_time_routine


if __name__ == "__main__":
    import doctest

    n_failuers, _ = doctest.testmod(
        optionflags=doctest.ELLIPSIS + doctest.NORMALIZE_WHITESPACE
    )
    if n_failuers > 0:
        raise ValueError(f"failed {n_failuers} tests")
