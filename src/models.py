"""
Replicates the Demographic Fiscal Model
(DFM) (see pp.121 Historical Dynamics, by
Peter Turchin, 2003) and the Demographic
Wealth Model (DFM) as stated in the 2024
paper (The Demographic-Wealth model for
cliodynamics) by Wittmann and Kuehn.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def k(S: float, init_k: int, c: int, init_s: float) -> float:
    """
    A state's carrying capacity. Determined
    in part by the state's current resources,
    it's initial carry capacity, how well
    the state can convert resources into
    carry capacity, and the state's initial
    resources.

    Parameters
    ----------
    S : float
        The state's accumulated resources.
    init_k : int
        The state's initial carry capacity.
    c : int
        How much the carry capacity can be
        increased.
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
    The Demographic Fiscal Model (DFM). Yields
    a state's population and accumulated
    resources as determined by its initial
    population, initial accumulated resources,
    rate of population growth, initial per
    capita taxation rate, maximum gain of
    increasing carrying capacity, and its
    per capita state expenditure.

    Parameters
    ----------
    t : int
        The current point in time.
    y : ArrayLike
        The current population and accumulated
        state resources
    args : tuple[float, float, float, float, float, float]
        The variables and parameters of the
        ODE system.

    Returns
    -------
    jax.Array
        The updated population and state
        resources.
    """
    N, S = y
    S = jnp.maximum(S, 0.0)
    r, init_p, beta, init_k, c, init_s = args
    dN = r * N * (1 - (N / k(S, init_k, c, init_s)))
    dS = (init_p * N * (1 - (N / k(S, init_k, c, init_s)))) - (beta * N)
    return jnp.array([dN, dS])


def DWM(
    t: int,
    y: ArrayLike,
    args: tuple[float, float, float, float, float, float, float, float],
) -> jax.Array:
    # get population and state resources
    N, S = y
    # ensure S >= 0
    S = jnp.maximum(S, 0.0)
    # get parameters
    r, init_p, beta, alpha, d, g, c, init_k = args
    # TODO: development equation components
    # update N and S
    dN = (r * N * (1 - (N / (init_k + (c * S))))) - (alpha * S * (N / (d + N)))
    dS = (g * S * N) - (beta * S)
    return jnp.array([dN, dS])
