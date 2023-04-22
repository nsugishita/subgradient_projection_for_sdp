# -*- coding: utf-8 -*-

"""Base class for configurations"""

import argparse
import copy as copy_module
import functools
import sys
import typing
import warnings
from difflib import SequenceMatcher

import numpy as np
import yaml


class BaseConfig(object):
    """Base class of config data structure"""

    def _keys(self):
        for key in self.__dict__.keys():
            if key.startswith("_"):
                continue
            yield key

    def _items(self):
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            yield key, value

    def _copy(self, **kwargs):
        """Create a deepcopy of self

        This creates a deepcopy of self. If keyword arguments
        are given, they are used to update the copied config.
        """
        return copy(self, **kwargs)

    def _update_from_dict(
        self, d: typing.Optional[typing.Dict[str, typing.Any]], inplace=False
    ) -> "BaseConfig":
        """Update config using a dict

        This creates an updated config using a given dict.
        If inplace=False (default), this creates a new config instance
        and returns it. Otherwise, this updates the config inplace.

        Parameters
        ----------
        d : dict
        inplace : bool, default False
        """
        if inplace:
            out = self
        else:
            out = self._copy()
        if d is None:
            return out
        valid_keys = list(out._keys())
        for key in d:
            if key not in valid_keys:
                raise KeyError(key)
        for key, value in d.items():
            setattr(out, key, value)
        return out

    def _asdict(self, *args, **kwargs) -> typing.Dict[str, typing.Any]:
        """Return a dict containing the configations

        Examples
        --------
        >>> c = DemoConfig()
        >>> c.tol = 0
        >>> c.time_limit = 40
        >>> c._asdict(only_modified=True)
        {'tol': 0, 'time_limit': 40}
        >>> c._asdict(only_modified=True, with_default=True)
        {'tol': (0.01, 0), 'time_limit': (0, 40)}
        >>> c._asdict(only_modified=True, shorten=True)
        {'tol': 0, 'time': 40}

        Returns
        -------
        dct : dict[str, tuple[Any, Any]] or dict[str, Any]
        """
        tpls = self._astuple(*args, **kwargs)

        if len(tpls) == 0:
            return {}

        if len(tpls[0]) > 2:
            out = {x[0]: x[1:] for x in tpls}
        else:
            out = {x[0]: x[1] for x in tpls}

        return out

    def _astuple(
        self, only_modified=False, with_default=False, shorten=False
    ) -> typing.Tuple:
        """Return a tuple

        Examples
        --------
        >>> c = DemoConfig()
        >>> c.tol = 0
        >>> c.time_limit = 40
        >>> c._astuple(only_modified=True)
        (('tol', 0), ('time_limit', 40))
        >>> c._astuple(only_modified=True, with_default=True)
        (('tol', 0.01, 0), ('time_limit', 0, 40))
        >>> c._astuple(only_modified=True, shorten=True)
        (('tol', 0), ('time', 40))

        Parameters
        ----------
        only_modified : bool, default False
            If True, only return items which are not default.
        with_default : bool, default False
            If True, return a tuple of
            (config_item_name, default_value, current_value).
            Otherwise, return a tuple of (config_item_name, current_value).

        Returns
        -------
        items : tuple[tuple[str, Any, Any], ...] or tuple[tuple[str, Any], ...]
        """
        out: typing.List[typing.Any] = []

        default = self.__class__()

        if shorten:
            key_map = self._shorten_name()
        else:
            key_map = {k: k for k in self._keys()}

        def eq(a, b):
            if isinstance(a, str) or isinstance(b, str):
                return a == b
            if np.isnan(a) and np.isnan(b):
                return True
            return a == b

        for key, value in self._items():
            if only_modified and eq(getattr(default, key, {}), value):
                continue
            if with_default:
                out.append(
                    (key_map[key], getattr(default, key, None), value),
                )
            else:
                out.append(
                    (key_map[key], value),
                )

        return tuple(out)

    def _asstr(self, *args, **kwargs) -> str:
        """Yield config items changed from their default values

        Examples
        --------
        >>> c = DemoConfig()
        >>> c.tol = 1e-3
        >>> c.time_limit = 40
        >>> c._asstr(only_modified=True, shorten=True)
        'tol_0.001_time_40'

        Returns
        -------
        res : str
        """
        res = []
        iter = self._astuple(*args, **kwargs)
        for key, new in iter:
            res.append(str(key))
            res.append(str(new))
        out = "_".join(res)
        for key in "~!@#$%^&*()`={}|[]\\;:'\"<>?,/":
            out = out.replace(key, "_")
        return out

    def _display_non_default(self, write=print) -> None:
        """Display config items changed from their default values

        Examples
        --------
        >>> c = DemoConfig()
        >>> c.tol = 0
        >>> c.time_limit = 40
        >>> c._display_non_default()
        tol        : 0.01 -> 0
        time_limit : 0    -> 40

        Yield
        -----
        key : str
        old : object
        new : object
        """
        _non_default = list(
            self._astuple(with_default=True, only_modified=True)
        )
        keys = [str(x[0]) for x in _non_default]
        olds = [str(x[1]) for x in _non_default]
        news = [str(x[2]) for x in _non_default]
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

    def _add_arguments(self, parser, conflict="warn", _stacklevel=2):
        """Add arguments to a given ArgumentParser

        Examples
        --------
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--foo', type=int)
        >>> config = DemoConfig()
        >>> config._add_arguments(parser)
        >>> args = parser.parse_args(['--foo', '10', '--step-size', '2e-1'])
        >>> args.step_size
        0.2

        Parameters
        ----------
        parser : argparse.ArgumentParser
        """
        assert conflict in ["pass", "raise", "warn"]
        conflict_names = []
        parser.add_argument("--config", type=str)
        for key, value in self._items():
            command_line_key = "--" + key.replace("_", "-")
            try:
                parser.add_argument(
                    command_line_key,
                    type=type(value),
                    dest=key,
                )
            except argparse.ArgumentError:
                conflict_names.append(command_line_key)
        if conflict_names:
            if conflict == "pass":
                return
            if len(conflict_names) > 1:
                msg = "conflicting option strings " + ", ".join(conflict_names)
            else:
                msg = "conflicting option string " + ", ".join(conflict_names)
            if conflict == "raise":
                raise ValueError(msg)
            elif conflict == "warn":
                warnings.warn(msg, stacklevel=3)

    def _parse_args(
        self,
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
        >>> config = DemoConfig()
        >>> config._add_arguments(parser)
        >>> args = parser.parse_args(['--foo', '10', '--step-size', '2e-1'])
        >>> config._parse_args(args)
        >>> config._display_non_default()
        step_size : 0.001 -> 0.2

        Parameters
        ----------
        args : argparse.Namespace or list[str], optional
            Arguments to be parsed. If not given, `sys.argv` is used.
        """

        if (args is None) or isinstance(args, list):
            if args is None:
                args = sys.argv[1:]

            parser = self._create_argparser()
            parsed, unknown = parser.parse_known_args(args)

        elif isinstance(args, argparse.Namespace):
            parsed = args

        else:
            raise TypeError(type(args))

        if (parsed.config is not None) and (len(parsed.config) > 0):
            with open(parsed.config, "r") as f:
                loaded = yaml.safe_load(f)
            self._update_from_dict(loaded, inplace=True)
        for key in self._keys():
            if key.startswith("_"):
                continue
            value = getattr(parsed, key)
            if value is None:
                continue
            setattr(self, key, value)

    def _create_argparser(self):
        parser = argparse.ArgumentParser()
        self._add_arguments(parser)
        return parser

    def _shorten_name(self):
        """Create a map to shorten config item names

        Returns
        -------
        map : dict[str, str]
        """
        keys = list(self._keys())
        out = {}

        for key in keys:
            if key.startswith("_"):
                continue

            out[key] = key

            if "_" in key:
                min_length = key.index("_") - 1
            else:
                min_length = len(key)

            for i in range(min_length, len(key)):
                shortened = key[: i + 1]
                status = True
                for k in keys:
                    if k == key:
                        continue
                    if k.startswith(shortened):
                        status = False
                        break
                if status:
                    out[key] = shortened
                    break

        return out


def freeze_attributes(cls):
    """Decorator to prevent new attribute assignments.

    A decorated class does not accept new attribute assignments
    after __init__.
    One can call 'unfreeze' to temporally allow new
    attribute and then call 'freeze_attributes' to protect again.

    Examples
    --------
    >>> @freeze_attributes
    ... class A:
    ...     def __init__(self):
    ...         self.name = 'foo'

    Instances of A do not accept new attributes, while
    editing existing attributes are allowed.

    >>> a = A()
    >>> a.name = 'bar'
    >>> a.age = 10
    Traceback (most recent call last):
     ...
    AttributeError: 'age' not found. did you mean 'name'?

    Use `unfreeze` to assign new attributes.

    >>> with unfreeze(a):
    ...     a.age = 10
    >>> a.age = 20

    When we have a class derived from A, we need to put the decorator to the
    deriving class too (to be more specific, we only need to use
    freeze_attributes decorator on the driving class only).

    >>> @freeze_attributes
    ... class B(A):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.height = 10.2
    >>> b = B()
    >>> b.name = 'baz'
    >>> b.height = 13.0
    >>> b.age = 40
    Traceback (most recent call last):
     ...
    AttributeError: 'age' not found. did you mean 'name'?

    """

    original_init = cls.__init__

    @functools.wraps(cls.__init__)
    def init(self, *args, **kwargs):
        # if we have inherited classes and if we put freeze_attributes
        # decorate on each of them, every time __init__ is called, this
        # routine is called. In the derived class, _frozen is not defined,
        # so original_frozen becomes True. In the super classes, _frozon
        # is defined and set to False, so original_frozen becomes False.
        # When we exit each __init__, _frozen is set to original_frozen
        # That is, only when we exit __init__ of the derived class,
        # we set _frozen to be True.
        original_frozen = getattr(self, "_frozen", True)
        self._frozen = False
        original_init(self, *args, **kwargs)
        self._frozen = original_frozen

    cls.__init__ = init

    original_setattr = cls.__setattr__

    def find_attribute_of_similar_name(set, key):
        def func(x):
            return similar(key, x)

        maybe = max(set, key=func)
        if similar(key, maybe) > 0.5:
            return maybe
        else:
            return None

    @functools.wraps(cls.__setattr__)
    def setattr(self, key, value):
        if getattr(self, "_frozen", False) and not hasattr(self, key):
            similar_attr = find_attribute_of_similar_name(
                self.__dict__.keys(), key
            )
            if similar_attr:
                raise AttributeError(
                    f"'{key}' not found. did you mean '{similar_attr}'?"
                )
            else:
                raise AttributeError(f"'{key}' not found.")
        original_setattr(self, key, value)

    cls.__setattr__ = setattr

    def _freeze(self, mode=True):
        """Prevent new item assignments."""
        self._frozen = mode

    def _unfreeze(self):
        """Allow new item assignments."""
        return unfreeze(self)

    cls._freeze = _freeze
    cls._unfreeze = _unfreeze

    return cls


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


class FreezeContext:
    """Context responsible to freeze an object when exiting."""

    def __init__(self, o, original_mode: bool) -> None:
        """Initialize an FreezeContext instance."""
        self.o = o
        self.original_mode = original_mode

    def __enter__(self, *args, **kwargs) -> None:
        """Enter the context."""
        pass

    def __exit__(self, *args, **kwargs) -> None:
        """Freeze the given object and exit the context."""
        self.o._freeze(mode=self.original_mode)


def freeze(o) -> FreezeContext:
    """Prevent new item assignments on a unfrozen instance or dict."""
    original_mode = o._frozen
    o._frozen = True
    return FreezeContext(o, original_mode)


def unfreeze(o) -> FreezeContext:
    """Allow new item assignments on a frozen instance or dict."""
    original_mode = o._frozen
    o._frozen = False
    return FreezeContext(o, original_mode)


@freeze_attributes
class DemoConfig(BaseConfig):
    """Data structure for configurations"""

    def __init__(self) -> None:
        """Initialise a DemoConfig instance"""
        self.verbose: int = 0

        self.problem_name: str = ""
        self.solver: str = ""

        # Suboptimality tolerance. See 'termination_criteria'.
        self.tol: float = 1e-2

        self.feas_tol: float = 1e-3

        # Time limit
        self.time_limit: float = 0

        # Iteration limit
        self.iteration_limit: float = 0

        # Parameter to control the regularization strength
        self.step_size: float = 1e-3


def copy(config, **kwargs):
    """Create a deepcopy

    This creates a deepcopy. If keyword arguments
    are given, they are used to update the copied config.
    """
    out = copy_module.deepcopy(config)
    out._update_from_dict(kwargs, inplace=True)
    return out


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
