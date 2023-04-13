# -*- coding: utf-8 -*-

"""Data structure for configurations"""

import argparse
import copy
import sys
import typing

import numpy as np


class Config(object):
    """Data structure for configurations"""

    def __init__(self) -> None:
        """Initialise a Config instance"""
        self.verbose: int = 0

        # Suboptimality tolerance. See 'termination_criteria'.
        self.tol: float = 1e-2

        # The condition to terminate solver.
        # {'lb_and_solution', 'solution', 'lb'}
        self.termination_criteria: str = "solution"

        # Time limit
        self.time_limit: float = 0

        # Iteration limit
        self.iteration_limit: float = 0

        # Frequency of log
        self.log_every: int = 1

        # Parameter to control the regularization strength
        self.step_size: float = 1e-3

        # Method to solve RMP model
        self.rmp_solver_method: str = "dual_simplex"

        # Type of initial cuts. {'lmi', 'linear', 'none'}
        self.initial_cut_type: str = "lmi"

        # Number of cuts corresponding to negative eigenvalues added to RMP.
        self.n_linear_cuts_for_unregularised_rmp: int = 1
        self.n_linear_cuts_for_regularised_rmp: int = 1

        self.eigenvector_combination_cut: int = 0

        self.n_lmi_cuts_for_unregularised_rmp: int = 0
        self.n_lmi_cuts_for_regularised_rmp: int = 0

        self.eval_lb_every: int = 1

        self.switch_to_cg_patience: int = -1

        self.use_dual_to_check_active_cut: int = 0

        self.log_eigenvectors: int = 0

    def copy(self, **kwargs) -> "Config":
        """Create a deepcopy of self

        This creates a deepcopy of self. If keyword arguments
        are given, they are used to update the copied config.
        """
        out = copy.deepcopy(self)
        out._update_from_dict_inplace(kwargs)
        return out

    def asdict(self) -> typing.Dict[str, typing.Any]:
        """Return a dict containing the configations"""
        return dict(self.__dict__)

    def non_default(
        self,
    ) -> typing.Iterator[typing.Tuple[str, typing.Any, typing.Any]]:
        """Yield config items changed from their default values

        Examples
        --------
        >>> c = Config()
        >>> c.tol = 0
        >>> c.time_limit = 40
        >>> list(c.non_default())
        [('tol', 0.01, 0), ('time_limit', 0, 40)]

        Yield
        -----
        key : str
        old : object
        new : object
        """
        default = Config().__dict__

        def eq(a, b):
            if isinstance(a, str) or isinstance(b, str):
                return a == b
            if np.isnan(a) and np.isnan(b):
                return True
            return a == b

        for key, value in self.__dict__.items():
            if not eq(default.get(key, {}), value):
                yield key, default.get(key, None), value

    def display_non_default(self, write=print) -> None:
        """Display config items changed from their default values

        Examples
        --------
        >>> c = Config()
        >>> c.tol = 0
        >>> c.time_limit = 40
        >>> c.display_non_default()
        tol        : 0.01 -> 0
        time_limit : 0    -> 40

        Yield
        -----
        key : str
        old : object
        new : object
        """
        non_default = list(self.non_default())
        keys = [str(x[0]) for x in non_default]
        olds = [str(x[1]) for x in non_default]
        news = [str(x[2]) for x in non_default]
        if len(keys) == 0:
            return
        width1 = max(len(x) for x in keys)
        width2 = max(len(x) for x in olds)
        formatter = (
            "{key:"
            + str(width1)
            + "s} : {old:"
            + str(width2)
            + "s} -> {new:s}"
        )
        for key, old, new in zip(keys, olds, news):
            write(formatter.format(key=key, old=old, new=new))

    def read_command_line_args(
        self, args: typing.Optional[typing.List[str]] = None
    ) -> typing.List[str]:
        """Read command line arguments and return unparsed arguments

        Parameters
        ----------
        args : list[str], optional
            Arguments to be parsed. If not given, `sys.argv`
            is used.

        Returns
        -------
        unknown : list[str]
            Arguments which are not used to update config.
        """
        if args is None:
            args = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str)
        for key, value in self.__dict__.items():
            command_line_key = "--" + key.replace("_", "-")
            parser.add_argument(
                command_line_key,
                type=type(value),
                dest=key,
            )
        parsed, unknown = parser.parse_known_args()
        if parsed.config is not None:
            import yaml

            with open(parsed.config, "r") as f:
                loaded = yaml.safe_load(f)
            self._update_from_dict_inplace(loaded)
        for key in self.__dict__:
            if key.startswith("_"):
                continue
            value = getattr(parsed, key)
            if value is None:
                continue
            self.__dict__[key] = value
        return unknown

    def _update_from_dict_inplace(
        self, d: typing.Optional[typing.Dict[str, typing.Any]]
    ) -> None:
        if d is None:
            return
        valid_keys = list(self.__dict__.keys())
        for key in d:
            if key not in valid_keys:
                raise KeyError(key)
        for key, value in d.items():
            self.__dict__[key] = value


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
