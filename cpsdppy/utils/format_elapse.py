# -*- coding: utf-8 -*-

"""Elapse formatter"""


def format_elapse(elapse: float) -> str:
    """Given elapse in seconds, return a formatted string

    Examples
    ---------
    >>> print(format_elapse(74))
    00:01:14
    >>> print(format_elapse(8493))
    02:21:33

    Parameters
    ----------
    elapse : float
        E.g., a value obtained as a difference of time.perf_counter().

    Returns
    -------
    text : str
        Formatted text, which looks like 'hh:mm:ss'.
    """
    elapse_h = int(elapse // (60 * 60))
    elapse = elapse - elapse_h * 60 * 60
    elapse_m = int(elapse // 60)
    elapse = elapse - elapse_m * 60
    elapse_s = int(elapse)
    return f"{elapse_h:02d}:{elapse_m:02d}:{elapse_s:02d}"


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
