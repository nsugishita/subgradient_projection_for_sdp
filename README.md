This is a python program for polynomial optimization.

# Install

## Virtual Env

First set up a virtual environment:

```
python3 -m venv env
. ./env/bin/activate
```

## Mosek

Get a lisence from Mosek [website](https://www.mosek.com).

## Gurobi

Gurobi is only required to run the column generation.
First, download Gurobi from the [website](https://www.gurobi.com).
Then, type the following commands:

```
pushd /Library/gurobi1001/macos_universal2/
python setup.py install
popd
```

## Insall Python Package

To install the package type

```
pip install -e .
```

## Set up pre-commit

Finally, set up pre-commit by the following command

```
pre-commit install
```
