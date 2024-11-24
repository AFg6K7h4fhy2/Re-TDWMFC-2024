"""
This file contains code that the author used
to understand and begin working on the
implementation of the Demographic Wealth
Model in Python.
"""

# %% IMPORTS

import jax.numpy as jnp

# %% PARAMETERS FOR TURCHIN MODEL

init_N = 0.5  # population
init_S = 0.0  # accumulated state resources
init_k = 1  # carry capacity of stateless population
r = 0.02
init_p = 1
c = 3
init_s = 10
beta = 0.25

# %% TURCHIN DFM


def k(init_k, c, init_s, S):
    """
    State carrying capacity.
    """
    return init_k + (c * (S / (init_s + S)))


def DFM(t, y, args):
    N, S = y
    dN = (r * N) * (1 - (N / k(S)))
    dS = init_p * N * (1 - (N / k(S)))
    return jnp.array([dN, dS])


# %% PARAMETERS
