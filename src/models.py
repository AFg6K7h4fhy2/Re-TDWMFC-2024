"""
Replicates the Demographic Fiscal Model
(DFM) (see pp.121 Historical Dynamics, by
Peter Turchin, 2003) and the Demographic
Wealth Model (DFM) as stated in the 2024
paper (The Demographic-Wealth model for
cliodynamics) by Wittmann and Kuehn.
"""

import jax.numpy as jnp
from jax.typing import Array

# def k(S: int, init_s: float, c: int, init_k: float) -> float:
#     """
#     State carrying capacity.
#     """
#     return init_k + (c * (S / (init_s + S)))


def k(S: int) -> int:
    return S


def DFM(t: int, y: Array, args: tuple[float, float, float]) -> Array:
    # get population and state resources
    N, S = y
    # ensure S >= 0
    S = jnp.maximum(S, 0.0)
    # get parameters
    r, init_p, beta = args
    # update N and S
    dN = r * N * (1 - (N / k(S)))
    dS = (init_p * N * (1 - (N / k(S)))) - (beta * N)
    return jnp.array([dN, dS])


def DWM(
    t: int,
    y: Array,
    args: tuple[float, float, float, float, float, float, float, float],
) -> Array:
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
