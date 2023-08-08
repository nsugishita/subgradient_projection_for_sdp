# -*- coding: utf-8 -*-

"model" "Run doc tests" ""

import doctest
import itertools
import os
import unittest

try:
    import cplex  # noqa: F401

    has_cplex = True
except ImportError:
    has_cplex = False


def load_tests(loader, tests, ignore):
    iterable = itertools.chain(os.walk("cpsdppy"), os.walk("examples"))
    for dirpath, dirnames, filenames in iterable:
        if "__pycache__" in dirpath:
            continue
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext == ".py":
                if (not has_cplex) and ("cplex" in filename):
                    continue
                _dirpath = dirpath.replace("/", ".")
                tested = _dirpath + "." + os.path.splitext(filename)[0]

                tests.addTests(
                    doctest.DocTestSuite(
                        tested,
                        optionflags=doctest.NORMALIZE_WHITESPACE
                        + doctest.ELLIPSIS
                        + doctest.IGNORE_EXCEPTION_DETAIL,
                    )
                )
            elif ext.lower() in [".markdown", ".md"]:
                tested = "../" + dirpath + "/" + filename

                tests.addTests(
                    doctest.DocFileSuite(
                        tested,
                        optionflags=doctest.NORMALIZE_WHITESPACE
                        + doctest.ELLIPSIS
                        + doctest.IGNORE_EXCEPTION_DETAIL,
                    )
                )

    return tests


if __name__ == "__main__":
    unittest.main()
