This is the official implementation of "Subgradient Projection Method with
Outer Approximation for Solving Semidefinite Programming Problems" by
Nagisa Sugishita and Miguel Anjos.

This is tested on Python 3.9.

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

## Gurobi, Mosek and SDPNAL+

Get a lisence from Gurobi [website](https://www.gurobi.com) and
Mosek [website](https://www.mosek.com).

Optionally, download SDPNAL+ and store it in `external` directory.
For more detail, see `external/README.md`.

## Set Up Virtual Environments and Data

To set up virtual environments for Python and julia and download data,
run the following command.

```
bash ./scripts/set_up.sh
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
