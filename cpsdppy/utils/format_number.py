# -*- coding: utf-8 -*-

"""Floating number formatter"""


def format_number(
    value, width=None, precision=None, return_scientific_flag=False
):
    """Given elapse in seconds, return a formatted string

    Examples
    ---------
    >>> format_number(74.405, width=9)
    '    74.41'
    >>> format_number(0.56, width=9)
    '     0.56'
    >>> format_number(-0.0008501, width=9)
    '-8.50e-04'
    >>> format_number(100939, width=9)
    ' 1.01e+05'

    Parameters
    ----------

    Returns
    -------
    text : str
        Formatted text, which looks like 'hh:mm:ss'.
    """
    if (width is None) and (precision is None):
        width = 9
        precision = 2
    if width is None:
        width = precision + 7
    if precision is None:
        precision = width - 7
    non_scientific_precision = precision
    if non_scientific_precision > 2:
        test = 0.02 < abs(value) < 999
    else:
        test = 0.2 < abs(value) < 999
    if test:
        format = "{:" + str(width) + "." + str(non_scientific_precision) + "f}"
        if return_scientific_flag:
            return format.format(value), False
        else:
            return format.format(value)
    else:
        format = "{:" + str(width) + "." + str(precision) + "e}"
        if return_scientific_flag:
            return format.format(value), True
        else:
            return format.format(value)


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
