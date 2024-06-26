# -*- coding: utf-8 -*-

"""Data structure for configurations"""

import argparse
import typing

from cpsdppy.utils import config as config_utils


@config_utils.freeze_attributes
class Config(config_utils.BaseConfig):
    """Data structure for configurations"""

    def __init__(self) -> None:
        """Initialise a Config instance"""
        self.verbose: int = 0

        self.problem_name: str = ""
        self.solver: str = ""

        # Currently only used in Mosek
        self.n_threads: int = 1

        # Suboptimality tolerance. See 'termination_criteria'.
        self.tol: float = 0.0

        self.feas_tol: float = 1e-3

        # The condition to terminate solver.
        # {'lb_and_solution', 'solution', 'lb'}
        self.termination_criteria: str = "solution"

        # Time limit
        self.time_limit: float = 3600

        # Iteration limit
        self.iteration_limit: float = 0

        # Frequency of log
        self.log_every: int = 1

        # Parameter to control the regularization strength
        self.step_size: float = 1.0

        self.step_size_manager_version: int = 2

        self.step_size_manager_shift: float = 0.0
        self.step_size_manager_scale: float = 1.2

        self.solver_interface: str = "gurobi"

        # Method to solve RMP model
        self.rmp_solver_method: str = "dual_simplex"

        self.projection_after_feasibility_step: int = 1
        self.projection_after_optimality_step: int = 1

        self.add_cuts_after_optimality_step: int = 1

        # If 1, do not run optimality step in subgradient projection.
        self.feasibility_recovery: int = 0

        self.memory: int = 20
        self.cut_deletion_criterion: str = "activity"

        self.duplicate_cut_check: int = 1

        # Type of initial cuts. {'lmi', 'linear', 'none'}
        self.initial_cut_type: str = "none"

        # Number of cuts corresponding to negative eigenvalues added to RMP.
        self.n_linear_cuts: int = 0
        self.n_linear_cuts_for_unregularised_rmp: int = -1
        self.n_linear_cuts_for_regularised_rmp: int = -1

        self.lmi_cuts_from_unique_vectors: int = 1

        self.n_lmi_cuts: int = 0
        self.n_lmi_cuts_for_unregularised_rmp: int = -1
        self.n_lmi_cuts_for_regularised_rmp: int = -1

        self.eigen_comb_cut: int = 1

        self.eval_lb_every: int = 0

        # Used in Mosek
        self.log_to_stdout: int = 0
        # Used in Mosek
        self.log_to_logger: int = 1

        self.julia_path: str = ""

        # If 1, do not save any results.
        self.dry_run: int = 0


def add_arguments(parser, conflict="warn", _stacklevel=3):
    """Add arguments to a given ArgumentParser

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> _ = parser.add_argument('--foo', type=int)
    >>> add_arguments(parser)
    >>> args = parser.parse_args(['--foo', '10', '--step-size', '2e-1'])
    >>> args.step_size
    0.2

    Parameters
    ----------
    parser : argparse.ArgumentParser
    """
    config = Config()
    config._add_arguments(parser, conflict=conflict, _stacklevel=_stacklevel)


def parse_args(
    config,
    args: typing.Optional[
        typing.Union[argparse.Namespace, typing.List[str]]
    ] = None,
) -> None:
    """Read command line arguments and return unparsed arguments

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> _ = parser.add_argument('--foo', type=int)
    >>> config = Config()
    >>> config._add_arguments(parser)
    >>> args = parser.parse_args(['--foo', '10', '--step-size', '2e-1'])
    >>> parse_args(config, args)
    >>> config._display_non_default()
    step_size : 1.0 -> 0.2

    Parameters
    ----------
    config : BaseConfig
    args : argparse.Namespace or list[str], optional
        Arguments to be parsed. If not given, `sys.argv` is used.
    """
    config._parse_args(args)


copy = config_utils.copy


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
