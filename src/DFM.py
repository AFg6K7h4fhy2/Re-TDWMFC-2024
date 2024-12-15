"""
Replicates the Demographic Fiscal Model
(DFM) (pp.121, Historical Dynamics, 2003) by
Peter Turchin.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def k(S: float, init_k: int, c: int, init_s: float) -> float:
    """
    A state's carrying capacity as used in
    the DFM. Determined in part by the
    state's current resources, it's initial
    carry capacity, how well the state can
    convert resources into carrying capacity,
    and the state's initial resources.

    Parameters
    ----------
    S : float
        The state's accumulated resources.
    init_k : int
        The state's initial carrying capacity.
    c : int
        How well the state can convert
        resources into carrying capacity, i.e.
        k_max - k_init.
    init_s : int
        The state's initial resources.

    Returns
    -------
    float
        The state's carrying capacity.
    """
    return init_k + (c * (S / (init_s + S)))


def DFM(
    t: int, y: ArrayLike, args: tuple[float, float, float, int, int, float]
) -> jax.Array:
    """
    The Demographic Fiscal Model (DFM), which
    models a state's population and
    accumulated resources, as determined by
    its initial population and initial
    accumulated resources, along with its
    rate of population growth, per capita
    taxation rate, ability to convert resources
    into carrying capacity, and its
    per capita expenditures.

    Parameters
    ----------
    t : int
        The current point in time.
    y : ArrayLike
        The current population and accumulated
        state resources.
    args : tuple[float, float, float, float, float, float]
        The variables and parameters of the
        ODE system.

    Returns
    -------
    jax.Array
        The resultant population and state
        resources.
    """
    # population, state resources
    N, S = y
    # population growth rate, taxation rate,
    # expenditure rate, initial carrying
    # capacity, carrying capacity increase
    # capacity from state resources,
    # initial state resources
    r, init_rho, beta, init_k, c, init_s = args
    dN = jnp.where(
        S >= 0.0,
        r * N * (1 - (N / k(S, init_k, c, init_s))),
        r * N * (1 - (N / k(0.0, init_k, c, init_s))),
    )
    dS = jnp.where(
        S >= 0.0,
        (init_rho * N * (1 - (N / k(S, init_k, c, init_s)))) - (beta * N),
        0.0,
    )
    return jnp.array([dN, dS])
