"""
Sets up a Demographic Fiscal Model (DFM) or
Demographic Wealth Model (DWM) experiment,
where an experiments consists of model
results corresponding to variable and
parameter values, which are specified in a
configuration file. The model experiment
visualizations only cover the relationship
between Population (N) and State Resources
or Wealth (S), over time. This script is
meant to be run from within `./src`. This
setup file can easily reproduce
the figures: 01, 02, 03, 08, 09, 10, 11 of
(The Demographic-Wealth model for
cliodynamics, 2024).

To run w/ normal plots:
python3 exp_setup.py --config "fig_01.toml"
python3 exp_setup.py --config "fig_03.toml" --plot
python3 exp_setup.py --config "fig_01.toml" --plot --style_path "../assets/styles/general_AF.mplstyle"
python3 exp_setup.py --config "exp_fig_01.toml" --plot --style_path "../assets/styles/general_AF.mplstyle"
"""

import argparse
import datetime as dt
import itertools as it
import pathlib
import time

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import toml
from matplotlib.backends.backend_pdf import PdfPages

from DFM import DFM
from DWM import DWM
from utils import ensure_listlike

# parameters for model running that ought
# never to have multiple values defined
# for them in a configuration file
CONFIG_SPECS = ["t0", "t1", "dt0"]

# the variables (population and state
# resources, in case of DFM, or wealth, in
# case of DWM)
CONFIG_VARS = ["init_N", "init_S"]

# currently supported models
SUPPORTED_MODELS = ["DFM", "DWM"]
MODELS = {"DFM": DFM, "DWM": DWM}

# additional models can be added here
CONFIG_PARAMS = {
    # the parameters that the DFM model needs
    # and or accepts; the order must match
    # that in the DFM function i.e.
    # r, init_rho, beta, init_k, c, init_s = args
    "DFM": [
        "r",  # population growth rate
        "init_rho",  # taxation rate
        "beta",  # expenditure rate
        "init_k",  # initial carrying capacity
        "c",  # max_k - init_k
        "init_s",  # initial state resources
    ],
    # the parameters that the DFM model needs
    # and or accepts; the order must match
    # that in the DWM function i.e.
    # r, beta, alpha, d, g, c, init_k = args
    "DWM": [
        "r",  # population growth rate
        "beta",  # expenditure rate
        "alpha",  # ?
        "d",  # strength of negative feedback from S to N
        "g",  # tax rate times the fraction of surplus gained through investing/expanding
        "c",  # max_k - init_k
        "init_k",  # initial carrying capacity
    ],
}

# the LaTeX labels for different variables
# and parameters used across the DWM and DFM
# models
LABELS = {
    "init_N": r"$N_0$",
    "init_S": r"$S_0$",
    "init_rho": r"$\rho_0$",
    "init_s": r"$s_0$",
    "init_k": r"$k_0$",
    "max_k": r"$k_{\text{max}}$",
    "c": r"$c$",
    "r": r"$r$",
    "beta": r"$\beta$",
    "alpha": r"$\alpha$",
    "g": r"$g$",
    "d": r"$d$",
}


def load_and_validate_config(
    config_path: str,
) -> dict[str, str | float | int | list[int] | list[float]]:
    """
    Extract content specified in a TOML
    configuration file.

    Parameters
    ----------
    config_file : str
        The name of the config file.

    Returns
    -------
    dict[str, str | float | int | list[int] | list[float]]
        A dictionary of model specifications,
        parameters, and variables. The
        following parameters and variables
        are permitted to be lists: init_N,
        init_S, init_rho, init_s, init_k,
        max_k, c, r, beta.
    """
    # convert the config path to a pathlib.Path object
    config_path = pathlib.Path(config_path)

    # confirm the config file exists
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # attempt loading the TOML config file
    try:
        config = toml.load(config_path)
    except Exception as e:
        raise Exception(f"Error while loading TOML: {e}")

    # ensure that all loaded configuration entries are proper
    loaded_entries = list(config.keys())
    if "model" not in loaded_entries:
        raise ValueError(
            'There is currently no "model" key in the loaded configuration elements.'
        )

    model_specified = config["model"]
    if model_specified not in SUPPORTED_MODELS:
        raise ValueError(
            f"The specified model ({model_specified}) is not in the supported models: {SUPPORTED_MODELS}."
        )

    missing_model_vals = [
        val
        for val in CONFIG_SPECS + CONFIG_VARS + CONFIG_PARAMS[model_specified]
        if val not in loaded_entries
    ]
    if missing_model_vals:
        raise ValueError(
            f"The following values ({missing_model_vals}) are missing for the {model_specified} model."
        )

    # ensure all config entries are list-like
    vars_to_make_listlike = CONFIG_VARS + CONFIG_PARAMS[model_specified]
    for k, v in config.items():
        if not isinstance(v, list) and k in vars_to_make_listlike:
            config[k] = ensure_listlike(v)

    # return the validated configuration
    return config


def get_y0s(
    init_N: list[float] | list[int], init_S: list[float] | list[int]
) -> list[jax.Array]:
    y0s = [jnp.array(pair) for pair in list(it.product(init_N, init_S))]
    return y0s


def get_args(
    model_input: dict[str, list[int] | list[float]]
) -> list[jax.Array]:
    args = [
        jnp.array(group)
        for group in list(
            it.product(
                *list(model_input.values()),
            )
        )
    ]
    return args


def run_clio_model(
    t0: int,
    t1: int,
    dt0: int,
    model_name: str,
    y0s: list[jax.Array],
    args: list[jax.Array],
) -> list[jax.Array]:
    """
    Run a single cliodynamics model.
    """
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, t1 - t0))
    solver = diffrax.Tsit5()
    term = diffrax.ODETerm(MODELS[model_name])
    sols = [
        diffrax.diffeqsolve(
            term, solver, t0, t1, dt0, y0, args=arg, saveat=saveat
        )
        for y0 in y0s
        for arg in args
    ]
    return sols


def get_sols_and_entries(
    model_selected: str,
    config: dict[str, str | float | int | list[int] | list[float]],
    y0s: list[jax.Array],
    args: list[jax.Array],
    sols: list[jax.Array],
):
    # associate the correctly ordered variables
    # and parameters with indices
    model_vars_and_params = CONFIG_VARS + CONFIG_PARAMS[model_selected]
    len_model_vars_and_params = {
        k: len(config[k]) for k in model_vars_and_params
    }
    model_vars_and_params_indices = list(range(len(model_vars_and_params)))
    model_vars_and_params_w_indices = {
        k: i
        for i, k in zip(model_vars_and_params_indices, model_vars_and_params)
    }
    entries = [
        y0.tolist() + arg.tolist()
        for i, y0 in enumerate(y0s)
        for j, arg in enumerate(args)
    ]
    sols_and_entries = list(zip(sols, entries))
    return (
        len_model_vars_and_params,
        model_vars_and_params_w_indices,
        sols_and_entries,
    )


def save_experiments(
    model_input: dict[str, list[float] | list[int]],
    sols: list[jax.Array],
):
    pass


def plot_experiments(
    config_name: str,
    len_model_vars_and_params: dict[str, list[float] | list[int]],
    model_vars_and_params_w_indices: dict[str, list[float] | list[int]],
    sols_and_entries: list[jax.Array],
    model_selected: str,
    model_input: dict[str, list[float] | list[int]],
    output_path: str,
    style_path: str,
    separate_plots: bool,
    overwrite: bool,
) -> None:

    # get style to use; again, assuming the
    # code is being run from within ./src
    if style_path:
        style_path = pathlib.Path(style_path)
        if style_path.exists():
            plt.style.use(str(style_path))
        else:
            raise FileNotFoundError(f"Style file at {style_path} not found.")

    # ensure output directory exists
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # prepare output file
    current_date = dt.datetime.now().strftime("%Y%m%d_%H")
    config_name = pathlib.Path(config_name).stem
    file_name = f"{config_name}_{current_date}.pdf"
    file_path = output_path / file_name

    # if not overwrite, don't make file
    if file_path.exists() and not overwrite:
        print(f"File {file_path} already exists. Skipping.")
        return

    # create pdf to save multiple figures
    with PdfPages(file_path) as pdf:
        # if there is but one plot to make
        # from the configuration (single
        # value for all variables and
        # parameters)
        if all(
            len_model_vars_and_params[k] == 1
            for k in model_vars_and_params_w_indices.keys()
        ):
            print(sols_and_entries)
        # if at least one variable or parameter
        # has multiple values
        else:
            for k, v in model_vars_and_params_w_indices.items():
                # variables or parameters of length
                # one are taken into account below
                if len_model_vars_and_params[k] > 1:
                    exclude_index = model_vars_and_params_w_indices[k]
                    # NEED to sort groups before
                    # it.groupby; remember
                    # s_e[1][i] gets the ith entry of
                    # y0.tolist() + arg.tolist()
                    sorted_group_data = sorted(
                        sols_and_entries,
                        key=lambda s_e: tuple(
                            s_e[1][i]
                            for i in range(len(s_e[1]))
                            if i != exclude_index
                        ),
                    )
                    groups_for_k = [
                        list(group)
                        for _, group in it.groupby(
                            sorted_group_data,
                            key=lambda s_e: tuple(
                                s_e[1][i]
                                for i in range(len(s_e[1]))
                                if i != exclude_index
                            ),
                        )
                    ]
                    # plot the group on an individual figure
                    # this will only ever plot N or S
                    for i, group in enumerate(groups_for_k):
                        figure, axes = plt.subplots(nrows=1, ncols=2)
                        axes[0].set_title(
                            f"{model_selected}: Population Change"
                        )
                        axes[0].set_ylabel(r"$N$", rotation=90)
                        axes[0].set_xlabel("t")
                        type_S = (
                            "Resources"
                            if model_selected == "DFM"
                            else "Wealth"
                        )
                        axes[1].set_title(f"{model_selected}: State {type_S}")
                        axes[1].set_ylabel(r"$S$", rotation=90)
                        axes[1].set_xlabel("t")
                        for elt in group:
                            sol = elt[0]
                            N, S = sol.ys.T
                            timepoints = sol.ts
                            param_val = elt[1][
                                model_vars_and_params_w_indices[k]
                            ]
                            axes[0].plot(
                                timepoints.tolist(),
                                N.tolist(),
                                label=rf"{LABELS[k]}={round(param_val, 2)}",
                            )
                            axes[1].plot(
                                timepoints.tolist(),
                                S.tolist(),
                                label=rf"{LABELS[k]}={round(param_val, 2)}",
                            )
                        axes[0].legend()
                        axes[1].legend()
                        # limit setting must come
                        # after axes.plot()
                        axes[1].set_xlim(xmin=0)
                        axes[1].set_ylim(ymin=0)
                        axes[0].set_xlim(xmin=0)
                        axes[0].set_ylim(ymin=0)
                        # plt.show()
                        # fig = create_plot(model_name, y0s, args, [sols[idx]], style)
                        pdf.savefig(figure)  # save fig
                        plt.close(figure)  # close fig after saving


def main(parsed_args: argparse.Namespace) -> None:

    # get configuration file
    config = load_and_validate_config(config_path=parsed_args.config)

    # get model (DFM or DWM) specified
    model_selected = config["model"]

    # get model variable and parameter
    # input dictionary from config
    model_input_dict = {k: config[k] for k in CONFIG_PARAMS[model_selected]}

    # gets y0s and args for model
    y0s = get_y0s(init_N=config["init_N"], init_S=config["init_S"])
    input_args = get_args(model_input=model_input_dict)

    # run model combinations, getting the
    # run time as well
    start = time.time()
    sols = run_clio_model(
        t0=config["t0"],
        t1=config["t1"],
        dt0=config["dt0"],
        model_name=model_selected,
        y0s=y0s,
        args=input_args,
    )
    elapsed = time.time() - start
    print(
        f"Experiments Using {model_selected} Ran In:\n"
        f"{round(elapsed, 5)} Seconds.\n"
    )

    # TODO: comment
    (
        len_model_vars_and_params,
        model_vars_and_params_w_indices,
        sols_and_entries,
    ) = get_sols_and_entries(
        model_selected=model_selected,
        config=config,
        y0s=y0s,
        args=input_args,
        sols=sols,
    )

    # (possibly) plot model results
    if parsed_args.plot:
        plot_experiments(
            config_name=parsed_args.config,
            len_model_vars_and_params=len_model_vars_and_params,
            model_vars_and_params_w_indices=model_vars_and_params_w_indices,
            sols_and_entries=sols_and_entries,
            model_selected=model_selected,
            model_input=model_input_dict,
            output_path=parsed_args.output_path,
            style_path=parsed_args.style_path,
            separate_plots=parsed_args.separate_plots,
            overwrite=parsed_args.overwrite,
        )
    # (possibly) save model results
    if parsed_args.save:
        pass


if __name__ == "__main__":
    # setup and use argument parser for
    # command line arguments
    parser = argparse.ArgumentParser(
        description="The argparser for DFM or DWM experiments."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="The path to a configuration file to use.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot (and have saved) the results of the experiment.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Whether to save the results the numerical results of the experiment.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=".",
        help="The folder for saved output files (plot PDFs or numerical results). Defaults to saving in the current directory.",
    )
    parser.add_argument(
        "--style_path",
        type=str,
        default=None,
        help="The path to a style file to use for plotting.",
    )
    parser.add_argument(
        "--separate_plots",
        action="store_true",
        help="Whether to plot N and S separately.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing saved PDF figures or numerical outputs.",
    )
    # pass the output to main
    parsed_args = parser.parse_args()
    main(parsed_args)

# TODO
# change configuration pathing
# make get sols and entries
# edit plotting code to use get sols and entries
# edit plotting code to save multiple figs to same pdf
# edit plotting code to save single plot
# remove max_k from plotting code
