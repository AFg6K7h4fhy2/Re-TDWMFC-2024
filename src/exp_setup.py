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
import itertools as it
import pathlib
import time
from collections.abc import Sequence
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import toml
from jax.typing import ArrayLike

from models import DFM, DWM

CONFIG_ENTRIES = [
    "init_N",
    "init_S",
    "init_p",
    "init_s",
    "init_k",
    "max_k",
    "c",
    "r",
    "beta",
    "t0",
    "t1",
    "dt0",
]
NON_LISTLIKE_KEYS = ["t0", "t1", "dt0"]


def check_values_interval(
    values: list[int] | list[float],
    min_value: int | float,
    max_value: int | float,
) -> bool:
    # check that each entry in values is
    # either float or integer
    if not all(isinstance(value, (int, float)) for value in values):
        raise TypeError(
            f"All values must be either int or float; got {values}."
        )
    # check that the values are captured in
    # an appropriate range
    if all(min_value <= value <= max_value for value in values):
        return True
    else:
        raise ValueError(
            f"All values must be between {min_value} and {max_value}."
        )


def ensure_listlike(x: Any) -> Sequence[Any]:
    return x if isinstance(x, Sequence) else [x]


def load_and_validate_config(
    config_file: str,
) -> dict[str, float | int | list[int] | list[float]]:
    """
    Extract content specified in a TOML
    configuration file.

    Parameters
    ----------
    config_file : str
        The name of the config file.

    Returns
    -------
    dict[str, float | int | list[int] | list[float]]
        A dictionary of model specifications,
        parameters, and variables. The
        following parameters and variables
        are permitted to be lists: init_N,
        init_S, init_p, init_s, init_k,
        max_k, c, r, beta. There are certain
        conditions for these parameters that
        must be met.
    """
    # the established config location
    base_path = pathlib.Path("../config")
    config_path = base_path / config_file
    # confirm the config file exists
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    # attempt loading the toml config file
    try:
        config = toml.load(config_path)
    except Exception as e:
        raise Exception(f"Error while loading TOML: {e}")
    # ensure that the entries are subset of
    # required entries
    loaded_entries = list(config.keys())
    if not set(loaded_entries).issubset(set(CONFIG_ENTRIES)):
        diff = set(loaded_entries).difference(set(CONFIG_ENTRIES))
        raise ValueError(
            f"Foreign keys present in the config: {diff}.\nAccepted: {set(CONFIG_ENTRIES)}"
        )
    # if certain entries are missing, fill in
    # with default values
    if sorted(loaded_entries) != sorted(CONFIG_ENTRIES):
        default_config_path = pathlib.Path("../config/default.toml")
        if not default_config_path.is_file():
            raise FileNotFoundError(
                f"Default config file not found: {default_config_path}"
            )
        try:
            default_config = toml.load(default_config_path)
        except Exception as e:
            raise Exception(f"Error while loading default TOML: {e}")
        default_entries = list(default_config.keys())
        diff_default = set(default_entries).difference(set(loaded_entries))
        diff_dict = {
            k: default_config[k]
            for k in list(diff_default)
            if k not in loaded_entries
        }
        config = {**config, **diff_dict}
    # ensure all config entries are listlike
    # and then have valid values
    for k, v in config.items():
        if not isinstance(v, list) and k not in NON_LISTLIKE_KEYS:
            config[k] = ensure_listlike(v)
    # TODO: check t0, t1, dt0
    # TODO: check c, max_k, init_k
    # TODO: non-listlike initN and initS?
    check_values_interval(
        values=config["init_N"], min_value=0.01, max_value=5.0
    )
    check_values_interval(
        values=config["init_S"], min_value=0.0, max_value=5.0
    )
    check_values_interval(values=config["init_p"], min_value=1, max_value=4)
    check_values_interval(values=config["init_s"], min_value=1, max_value=30)
    check_values_interval(values=config["init_k"], min_value=1, max_value=10)
    check_values_interval(values=config["max_k"], min_value=1, max_value=10)
    check_values_interval(values=config["c"], min_value=1, max_value=10)
    check_values_interval(values=config["r"], min_value=0.01, max_value=0.90)
    check_values_interval(
        values=config["beta"], min_value=0.00, max_value=0.90
    )
    max_init_k = max(config["init_k"])
    max_max_k = max(config["max_k"])
    if max_init_k >= max_max_k:
        raise ValueError(
            f"Maximum carry capacity (got {max_max_k}) must be greater than initial carry capacity (got {max_init_k})."
        )
    print(config)
    return config


def run_model(
    config: dict[str, float | int | list[int] | list[float]], model: str
) -> tuple[list[jax.Array], list[jax.Array], list[jax.Array]]:
    """
    Run a single cliodynamics model (DFM or
    DWM) using a combination of variables and
    parameters.

    Parameters
    ----------
    config : dict[str, float | int | list[int] | list[float]]
        A dictionary of model specifications,
        parameters, and variables. The
        following parameters and variables
        are permitted to be lists: init_N,
        init_S, init_p, init_s, init_k,
        max_k, c, r, beta. There are certain
        conditions for these parameters that
        must be met.
    model : str
        The name of the model. Either DFM
        or DWM.

    Returns
    -------
    tuple[list[jax.Array], list[jax.Array], list[jax.Array]]
        The solutions, arguments, and initial
        variables for the experiments desired.
    """
    # get model times from config
    t0 = config["t0"]
    t1 = config["t1"]
    dt0 = config["dt0"]
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, t1 - t0))
    # choose solver
    solver = diffrax.Tsit5()
    # get appropriate model and args
    if model == "DFM":
        term = diffrax.ODETerm(DFM)
        args = [
            jnp.array(group)
            for group in list(
                it.product(
                    config["r"],
                    config["init_p"],
                    config["beta"],
                    config["init_k"],
                    config["c"],
                    config["init_s"],
                )
            )
        ]
    if model == "DWM":
        term = diffrax.ODETerm(DWM)
        args = [
            jnp.array(group)
            for group in list(
                it.product(
                    config["r"],
                    config["init_p"],
                    config["beta"],
                    config["alpha"],
                    config["d"],
                    config["g"],
                    config["c"],
                    config["init_k"],
                )
            )
        ]
    # get combinations of parameters, group
    # by each parameter
    y0s = [
        jnp.array(pair[0], pair[1])
        for pair in list(it.product(config["init_N"], config["init_S"]))
    ]
    sols = [
        diffrax.diffeqsolve(
            term, solver, t0, t1, dt0, y0, args=arg, saveat=saveat
        )
        for y0 in y0s
        for arg in args
    ]
    return (sols, args, y0s)


def plot_figure(
    sols: list[ArrayLike],
    args: list[ArrayLike],
    y0s: list[jax.Array],
    to_save: bool,
    save_name: str,
) -> None:
    pass


def main(args: argparse.Namespace) -> None:

    # get configuration file
    config = load_and_validate_config(config_file=args.config)

    # get model name
    model_name = "DFM" if args.DFM else "DWM"

    # get model results
    start = time.time()
    sols, ode_args, y0s = run_model(config=config, model=model_name)
    elapsed = time.time() - start
    print(
        f"Experiments Using {model_name} Ran In:\n{round(elapsed, 5)} Seconds.\n"
    )

    # # plot and (possibly) save
    # plot_figure(sols=sols, to_save=args.save_as_pdf, save_name=args.save_name)
    print(config, model_name)


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
        "--save_as_pdf",
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
