# -*- coding: utf-8 -*-

"""Parse str of ints or range"""

import numpy as np


def parse_ints(value, numpy: bool = False, atleast_1d: bool = True):
    """Parse str of ints or range

    This parses a string of int, ints or range.
    This may be useful to parse user input from command line.

    Examples
    --------
    >>> parse_ints('1')  # if a single number is given, it's simply parsed
    [1]
    >>> parse_ints('2, 10')  # multiple values separated by commas.
    [2, 10]
    >>> parse_ints('-3,1,-5,0')
    [-3, 1, -5, 0]
    >>> parse_ints('10:15')  # range can be specified with colons.
    range(10, 15)
    >>> parse_ints('0:10:2')  # one can specify the step
    range(0, 10, 2)
    >>> parse_ints(':2')  # if the start is omitted, 0 is used.
    range(0, 2)
    >>> parse_ints('5-20')  # dashes are interpretted as a colon.
    range(5, 20)
    >>> parse_ints('0:3.4')  # invalid values raises an ValueError
    Traceback (most recent call last):
      ...
    ValueError: 0:3.4

    Parameters
    ----------
    value : str
        Value to be parsed.
    numpy : bool, default False
        Return numpy array instead of python native objects.
    atleast_1d : bool, default True
        Return a list even if only single number is given.

    Returns
    -------
    parsed : int, list or array

    Raises
    ------
    ValueError
    """
    import re

    given = value
    ret: object
    if isinstance(value, (list, tuple)):

        def is_float(x):
            return isinstance(x, float)

        if any(is_float(x) for x in value):
            raise ValueError(given)
        ret = [int(x) for x in value]
        if numpy:
            return np.array(ret)
        else:
            return ret

    value = value.strip()
    if isinstance(value, str):
        value = re.sub(r"(\d)-(\d)", r"\g<1>:\g<2>", value)
    if value is None or value == "":
        return [] if atleast_1d else None
    if isinstance(value, (list, tuple)):
        if all([isinstance(x, int) for x in value]):
            return value
        else:
            raise ValueError(given)

    if isinstance(value, str):
        if value.startswith(":"):
            value = "0" + value

        if ":" in value:
            splitted = value.split(":")
            if any(s == "" for s in splitted):
                raise ValueError(given)
            if len(splitted) == 2:
                try:
                    start, stop = map(int, splitted)
                except ValueError:
                    pass
                else:
                    if numpy:
                        ret = np.arange(start, stop)
                    else:
                        ret = range(start, stop)
                    return ret
            elif len(splitted) == 3:
                try:
                    start, stop, step = map(int, splitted)
                except ValueError:
                    pass
                else:
                    if numpy:
                        ret = np.arange(start, stop, step)
                    else:
                        ret = range(start, stop, step)
                    return ret

        elif "," in value:
            splitted = value.split(",")
            try:
                ret = list(map(int, splitted))
            except ValueError:
                pass
            else:
                if numpy:
                    ret = np.array(ret)
                return ret

    try:
        parsed = int(value)
    except ValueError:
        pass
    else:
        if numpy and atleast_1d:
            return np.array([parsed])
        elif atleast_1d:
            return [parsed]
        else:
            return parsed

    raise ValueError(given)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
