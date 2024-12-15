"""
Replicates the Demographic Wealth
Model (DFM) (pp. 08, The Demographic-Wealth
model for cliodynamics, 2024) by Wittmann
and Kuehn.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def k(S: float, init_k: int, c: int) -> float:
    """
    A state's carrying capacity as used in
    the DWM. Determined in part by the
    state's current wealth, it's initial
    carry capacity, how well the state can
    convert wealth into carrying capacity,
    and the state's initial wealth.

    Parameters
    ----------
    S : float
        The state's wealth.
    init_k : int
        The state's initial carrying capacity.
    c : int
        How well the state can convert
        wealth into carrying capacity, i.e.
        k_max - k_init.

    Returns
    -------
    float
        The state's carrying capacity.
    """
    return init_k + (c * S)


def DWM(
    t: int,
    y: ArrayLike,
    args: tuple[float, float, float, float, float, float, float],
) -> jax.Array:
    """
    The Demographic Wealth Model (DWM), which
    models a state's population and
    wealth, as determined by
    its initial population and its
    rate of population growth, per capita
    taxation rate, fraction of surplus gained
    through investing/expanding (using its
    wealth), ability to convert resources
    into carrying capacity, per capita
    expenditures, and negative feedback
    between population and wealth.

    Parameters
    ----------
    t : int
        The current point in time.
    y : ArrayLike
        The current population and accumulated
        state wealth.
    args : tuple[float, float, float, float, float, float, float]
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
    # population growth rate,
    # expenditure rate,
    # initial state resources,
    # expenditure rate, negative interaction
    # strength between state wealth and
    # population size, tax rate times the
    # fraction of surplus gained through
    # investing/expanding, carrying capacity
    #  increase capacity from state wealth,
    # initial carrying capacity,
    r, beta, alpha, d, g, c, init_k = args
    dN = jnp.where(
        S >= 0.0,
        (r * N * (1 - (N / k(S, init_k, c)))) - (alpha * S * (N / (d + N))),
        (r * N * (1 - (N / k(0.0, init_k, c))))
        - (alpha * 0.0 * (N / (d + N))),
    )
    dS = jnp.where(S >= 0.0, (g * S * N) - (beta * S), 0.0)
    return jnp.array([dN, dS])
