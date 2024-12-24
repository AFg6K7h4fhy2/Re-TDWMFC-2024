"""
Utility functions for aiding with DFM or DWM
experiment setup.
"""

from collections.abc import Sequence
from typing import Any


def check_values_interval(
    values: list[int] | list[float],
    min_value: int | float,
    max_value: int | float,
) -> bool:
    """
    Checks whether all numerical elements of
    a list are within specified bounds.

    Parameters
    ----------
    values : list[int] | list[float]
        Model variables or parameters.
    min_value : int | float
        The lower bound (inclusive).
    max_value : int | float
        The upper bound (inclusive).

    Returns
    -------
    bool
        Whether all values are within the
        specified bounds.
    """
    # make sure all elements are int or float
    if not all(isinstance(value, (int, float)) for value in values):
        raise TypeError(
            f"All values must be either int or float; got {values}."
        )
    # make sure all elements are in bounds
    if all(min_value <= value <= max_value for value in values):
        return True
    else:
        raise ValueError(
            f"All values must be between {min_value} and {max_value}."
        )


def ensure_listlike(x: Any) -> Sequence[Any]:
    """
    Ensures that an element is listlike,
    i.e. a Sequence.

    Parameters
    ----------
    x : Any
        An object intended to be listlike.

    Returns
    -------
    Sequence[Any]
        The object if already listlike or
        a list containing the object.
    """
    return x if isinstance(x, Sequence) else [x]
