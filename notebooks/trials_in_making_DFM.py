"""
This file contains code that the author used
to understand and begin working on the
implementation of the Demographic Fiscal
Model (DFM) in Python, with initial values
and initial parameters values from the
paper (The Demographic-Wealth model for
cliodynamics).
"""

# %% IMPORTS

import diffrax
import jax.numpy as jnp

# %% INITIAL VALUES FOR DFM

init_N = 0.5  # initial population
init_S = 0.0  # accumulated state resources

# %% PARAMETERS FOR DFM

init_k = 1  # carry capacity of stateless population
r = 0.02  # rate of population growth
init_p = 1  # initial per capita taxation rate
max_k = 4  # maximum carrying capacity
c = max_k - init_k  # maximum gain of increasing carrying capacity
init_s = 10  # initial state resources (grain)
beta = 0.25  # per capita state expenditure

# %% STATE CARRYING CAPACITY


def k(S):
    """
    State carrying capacity.
    """
    return init_k + (c * (S / (init_s + S)))


# %% DEMOGRAPHIC FISCAL MODEL


def DFM(t, y, args):
    N, S = y  # get population and state resources
    r, init_p, beta = args  # get parameters
    dN = r * N * (1 - (N / k(S)))
    dS = init_p * N * (1 - (N / k(S))) - (beta * N)
    return jnp.array([dN, dS])


# %% GET THE DFM SOLUTION

term = diffrax.ODETerm(DFM)
solver = diffrax.Tsit5()
t0 = 0
t1 = 100
dt0 = 1
y0 = (init_N, init_S)
args = (r, init_p, beta)
saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 100))
sol = diffrax.diffeqsolve(
    term, solver, t0, t1, dt0, y0, args=args, saveat=saveat
)
