This is the official implementation of "Subgradient Projection Method with
Outer Approximation for Solving Semidefinite Programming Problems".

This is tested on Python 3.9, Julia 1.8.5 and g++ 13.0.1 20230405.

# Install

## Dependencies

This program requires `julia`, `g++` and `cmake` in `bin` directory.
Assuming you have these binaries on your machine, put symbolic links
in `bin` directory.

```
├─ README.md
├─ data
├─ bin
│   ├─ julia
│   ├─ g++
│   └─ cmake
.
.
└─ setup.py
```

## Virtual Env

First set up a virtual environment:

```
python3 -m venv env
. ./env/bin/activate
```

## Gurobi and Mosek

Get a lisence from Gurobi [website](https://www.gurobi.com) and
Mosek [website](https://www.mosek.com).

## Insall Python Package

To install the package type

```
pip install -e .
```

## Install COSMO

There is a script to install COSMO.
Type the following command.

```
./scripts/set_up_cosmo.sh
```

## Download Data

We need to download data to run the experiments.
Check the license of SDPLIB etc. and use the following commands.

```
pushd data
./download_sdplib.sh
./setup_julia_data.sh
popd
```

## Build C++ Extensions

Finally, we need to build extensions using the following commands.

```
./scripts/build.sh
```

# Run Experiments

Before running the program, we need to set up environment variables.
Typically the following script will handle these.

```
. ./scripts/activate.sh
```

Now, we are ready to run the program.
We can solve a simple 2D example:

```
python ./examples/solve_simple_2d_problem.py
```

or run the evaluation:

```
python ./examples/run_experiments.py
```
