"""
Sets up an experiment for use with the
Demographic Fiscal Model (DFM) or Demographic
Wealth Model (DWM) by reading in variable and
parameter values from configuration files.
This script must be ran from within the
folder `src`. To run:

python3 exp_setup.py --DFM --config "fig01.toml"
"""

import argparse
import pathlib
import time
from collections.abc import Sequence
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import toml
from jax.typing import ArrayLike

from models import DFM

CONFIG_GROUPS = ["model", "variables", "parameters"]


def read_and_load_config(
    config_path: pathlib.Path,
) -> dict[str, float | int | list[int] | list[float]]:
    """
    Extract content specified in a TOML
    configuration file.

    Parameters
    ----------
    file_path :
        X

    Returns
    -------

    """
    try:
        loaded_toml = toml.load(config_path)
    except Exception as e:
        raise Exception(f"Error while loading TOML: {e}")
    loaded_keys = list(loaded_toml.keys())
    if sorted(loaded_keys) != sorted(CONFIG_GROUPS):
        raise ValueError(
            f"Error: The keys in the TOML file are {loaded_keys}, but expected {CONFIG_GROUPS}."
        )
    return loaded_toml


def ensure_listlike(x: Any) -> Sequence[Any]:
    """
    Idea sourced from the following:
    https://stackoverflow.com/questions/66485566/
    """
    return x if isinstance(x, Sequence) else [x]


def run_model(
    config: dict[str, float | int | list[int] | list[float]], model: str
) -> jax.Array:
    print(config)
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
    return sol


def plot_figure(sols: ArrayLike, to_save: bool, save_name: str) -> None:
    pass


def main(args: argparse.Namespace) -> None:

    # get configuration file
    config = read_and_load_config(config_path=args.config)

    # get model name
    model_name = "DFM" if args.DFM else "DWM"

    # get model results
    start = time.time()
    sols = run_model(config=config, model=model_name)
    elapsed = time.time() - start
    print(f"Model {model_name} Ran In {elapsed} Seconds.")

    # plot and (possibly) save
    plot_figure(sols=sols, to_save=args.save, save_name=args.save_name)


if __name__ == "__main__":
    # setup and use argument parser
    parser = argparse.ArgumentParser(
        description="Argparser for re-tdwmfc-wittmann. Helps with which model to use, figure to re-create, and whether to save plots."
    )
    parser.add_argument(
        "--DFM",
        action="store_true",
        help="Whether to use the Demographic Fiscal Model (DFM).",
    )
    parser.add_argument(
        "--DWM",
        action="store_true",
        help="Whether to use the Demographic Wealth Model (DWM).",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="The name of the configuration file to use.",
    )
    parser.add_argument(
        "--style",
        type=str,
        help="(optional) The name of the style file to use. Defaults to Grayscale.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="(optional) Whether to save plots that were generated.",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="Default_Figure_Name",
        help="(optional) The name of the plot to save.",
    )
    args = parser.parse_args()
    # check that one but not both models
    # were provided
    if (args.DFM + args.DWM) != 1:
        raise ValueError(
            "You must provide exactly one of --DFM or --DWM, not both or neither."
        )
    # pass args to main and execute model
    main(args)
