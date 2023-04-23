# -*- coding: utf-8 -*-

import os
from typing import Dict

from setuptools import setup

package_name = "cpsdppy"

_locals: Dict = {}
with open(os.path.join(package_name, "_version.py")) as fp:
    exec(fp.read(), None, _locals)
__version__ = _locals["__version__"]

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name=package_name,
    version=__version__,
    description="Cutting-plane method for SDP",
    long_description=readme,
    author="Nagisa Sugishita",
    author_email="s1576972@ed.ac.uk",
    install_requires=[
        "gurobipy",
        "matplotlib",
        "Mosek",
        "numpy",
        "pandas",
        "pyyaml",
        "scipy",
        "Jinja2",
        # For development
        "black",
        "pre-commit",
        "mypy",
    ],
    license=license,
    packages=[package_name],
    include_package_data=True,
    test_suite="tests",
)
