"""
Sets up an experiment for use with the
Demographic Fiscal Model (DFM) or Demographic
Wealth Model (DWM) by reading in variable and
parameter values from configuration files.
"""

from collections.abc import Sequence

import diffrax
import jax.numpy as jnp

from models import DFM


def read_config():
    pass


def ensure_listlike(x: any):
    """
    Idea sourced from the following:
    https://stackoverflow.com/questions/66485566/
    """
    return x if isinstance(x, Sequence) else [x]


def plot_figure():
    pass


def run_model():
    term = diffrax.ODETerm(DFM)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = 500
    dt0 = 1
    init_p = 1
    r = 0.02
    init_N = 0.5
    init_S = 0.0
    y0 = jnp.array([init_N, init_S])
    args_01 = jnp.array([r, init_p, 0.0])
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, t1 - t0))
    sol = diffrax.diffeqsolve(
        term, solver, t0, t1, dt0, y0, args=args_01, saveat=saveat
    )
    print(sol)


def main():
    pass


if __name__ == "__main__":
    # parser =

    # pass args to main and run
    main()


# %%
