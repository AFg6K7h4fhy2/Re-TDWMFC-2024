"""
Sets up a Demographic Fiscal Model (DFM) or
Demographic Wealth Model (DWM) experiment.
Each experiment consists of model results
corresponding to variable and parameter values
specified in a configuration file. Visualization
of Population (N) and State Resources / Wealth
(S) over time or in relation to a model
variable or parameter is afforded as well.
This script is meant to be run from within
`./src`. This setup file can easily reproduce
the figures: 01, 02, 03, 08, 09, 10, 11.

To run w/ normal plots:
python3 exp_setup.py --config "fig_01.toml"
"""

import argparse
import itertools as it
import pathlib
import time

import diffrax
import jax
import jax.numpy as jnp
import toml

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

# assets folders
FIGURE_DIRECTORY = "../assets/figures/"
RESULTS_DIRECTORY = "../assets/output/"
STYLES_DIRECTORY = "../assets/styles/"

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
    config_file: str,
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
    # the established config location;
    # assumed one is running code in ./src
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
    # ensure that all loaded configuration
    # entries are proper
    loaded_entries = list(config.keys())
    if "model" not in loaded_entries:
        raise ValueError(
            'There is currently not "model" key in the loaded configuration elements.'
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
    # ensure all config entries are listlike
    vars_to_make_listlike = CONFIG_VARS + CONFIG_PARAMS[model_specified]
    for k, v in config.items():
        if not isinstance(v, list) and k in vars_to_make_listlike:
            config[k] = ensure_listlike(v)
    # ensure variables and parameters are
    # within the correct intervals
    # check_values_interval(
    #     values=config["init_N"], min_value=0.01, max_value=5.0
    # )
    # check_values_interval(
    #     values=config["init_S"], min_value=0.0, max_value=5.0
    # )
    # check_values_interval(values=config["init_rho"], min_value=1, max_value=4)
    # check_values_interval(values=config["init_s"], min_value=1, max_value=30)
    # check_values_interval(values=config["init_k"], min_value=1, max_value=10)
    # check_values_interval(values=config["max_k"], min_value=1, max_value=10)
    # check_values_interval(values=config["c"], min_value=1, max_value=10)
    # check_values_interval(values=config["r"], min_value=0.01, max_value=0.90)
    # check_values_interval(
    #     values=config["beta"], min_value=0.00, max_value=0.90
    # )
    # max_init_k = max(config["init_k"])
    # min_max_k = min(config["max_k"])
    # if max_init_k >= min_max_k:
    #     raise ValueError(
    #         f"Minimum max carry capacity (got {min_max_k}) must be greater than initial carry capacity (got {max_init_k})."
    #     )
    return config


def get_y0s(
    init_N: list[float] | list[int], init_S: list[float] | list[int]
) -> list[jax.Array]:
    """
    TODO:
    """
    y0s = [jnp.array(pair) for pair in list(it.product(init_N, init_S))]
    return y0s


def get_args(
    model_input: dict[str, list[int] | list[float]]
) -> list[jax.Array]:
    """
    TODO:
    """
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


# def create_plot(
#     model_name: str,
#     y0s: list[jax.Array],
#     args: list[jax.Array],
#     sols: list[jax.Array],
#     style: str = None,
# ) -> plt.Figure:
#     """
#     Creates a plot for the model results based on the provided data.

#     Parameters
#     ----------
#     model_name : str
#         The name of the model (e.g., DFM or DWM).
#     y0s : list[jax.Array]
#         The initial variable values for the model.
#     args : list[jax.Array]
#         Different combinations of parameters for the specified model.
#     sols : list[jax.Array]
#         A list of ODE solutions corresponding to different combinations of variables and parameters of the specified model.
#     style : str
#         Optional style to apply to the plot (e.g., 'seaborn', 'ggplot').

#     Returns
#     -------
#     figure : matplotlib.figure.Figure
#         The generated figure object for the plot.
#     """
#     figure, axes = plt.subplots(nrows=1, ncols=2)
#     # population side plotting
#     axes[0].set_title(f"{model_name}: Population Change")
#     axes[0].set_ylabel(r"$N$", rotation=90)
#     axes[0].set_xlabel("t")
#     type_S = "Resources" if model_name == "DFM" else "Wealth"
#     axes[1].set_title(f"{model_name}: State {type_S}")
#     axes[1].set_ylabel(r"$S$", rotation=90)
#     axes[1].set_xlabel("t")
#     for elt in group:
#         sol = elt[0]
#         N, S = sol.ys.T
#         timepoints = sol.ts
#         param_val = elt[1][model_vars_and_params_w_indices[k]]
#         axes[0].plot(
#             timepoints.tolist(),
#             N.tolist(),
#             label=rf"{LABELS[k]}={round(param_val, 2)}",
#         )
#         axes[1].plot(
#             timepoints.tolist(),
#             S.tolist(),
#             label=rf"{LABELS[k]}={round(param_val, 2)}",
#         )

#     axes[0].legend()
#     axes[1].legend()
#     # limit setting must come
#     # after axes.plot()
#     axes[1].set_xlim(xmin=0)
#     axes[1].set_ylim(ymin=0)
#     axes[0].set_xlim(xmin=0)
#     axes[0].set_ylim(ymin=0)
#     plt.show()


def save_experiments(
    model_input: dict[str, list[float] | list[int]],
    sols: list[jax.Array],
):
    pass


# def plot_experiments(
#     y0s: list[jax.Array],
#     args: list[jax.Array],
#     sols: list[jax.Array],
#     model_vars_and_params: dict[str, int | float | list[float] | list[int]],
#     len_model_vars_and_params: dict[str, int],
#     model_input: dict[str, list[float] | list[int]],
#     save_name: str,
#     style_name: str,
#     separate_plots: bool,
#     overwrite: bool,
# ) -> None:

#     # get style to use; again, assuming the
#     # code is being run from within ./src
#     base_style_path = pathlib.Path("../assets/styles")
#     if style is not None:
#         style_path = base_style_path / (style + ".mplstyle")
#         if style_path.exists():
#             plt.style.use(str(style_path))
#             print(f"Loaded style: {style}.")
#         else:
#             raise FileNotFoundError(f"Style file {style}.mplstyle not found.")

#     # by default, this function looks for
#     # ../assets/figures as a directory...
#     current_date = dt.datetime.now().isoformat()
#     if not os.path.exists(FIGURE_DIRECTORY):
#         os.makedirs(FIGURE_DIRECTORY)
#         raise FileNotFoundError(
#             f"The directory {FIGURE_DIRECTORY} does not exist."
#         )
#     file_name = f"{save_name}_{current_date}.pdf"
#     file_path = os.path.join(FIGURE_DIRECTORY, file_name)

#     # associate the correctly ordered variables
#     # and parameters with indices
#     model_vars_and_params_indices = list(range(len(model_vars_and_params)))
#     model_vars_and_params_w_indices = {
#         k: i
#         for i, k in zip(model_vars_and_params_indices, model_vars_and_params)
#     }
#     entries = [
#         y0.tolist() + arg.tolist()
#         for i, y0 in enumerate(y0s)
#         for j, arg in enumerate(args)
#     ]
#     sols_and_entries = list(zip(sols, entries))

#     # acceptable xaxis values
#     acceptable_xaxes = CONFIG_PARAMS + ["time"]
#     # check that

#     if not plot_as_img:
#         # create pdf to save multiple figures
#         with PdfPages(file_path) as pdf:
#             # for each figure, create
#             # the plot and add it to the PDF
#             # plot by groups of variables and parameters
#             for k, v in model_vars_and_params_w_indices.items():
#                 # variables or parameters of length
#                 # one are taken into account below
#                 if len_model_vars_and_params[k] > 1:
#                     exclude_index = model_vars_and_params_w_indices[k]
#                     # NEED to sort groups before
#                     # it.groupby!!!; remember
#                     # s_e[1][i] gets the ith entry of
#                     # y0.tolist() + arg.tolist()
#                     sorted_group_data = sorted(
#                         sols_and_entries,
#                         key=lambda s_e: tuple(
#                             s_e[1][i]
#                             for i in range(len(s_e[1]))
#                             if i != exclude_index
#                         ),
#                     )
#                     groups_for_k = [
#                         list(group)
#                         for _, group in it.groupby(
#                             sorted_group_data,
#                             key=lambda s_e: tuple(
#                                 s_e[1][i]
#                                 for i in range(len(s_e[1]))
#                                 if i != exclude_index
#                             ),
#                         )
#                     ]
#                     # plot the group on an individual figure
#                     # this will only ever plot N or S
#                     for i, group in enumerate(groups_for_k):
#                         figure, axes = plt.subplots(nrows=1, ncols=2)
#                         axes[0].set_title(f"{model_name}: Population Change")
#                         axes[0].set_ylabel(r"$N$", rotation=90)
#                         axes[0].set_xlabel("t")
#                         type_S = (
#                             "Resources" if model_name == "DFM" else "Wealth"
#                         )
#                         axes[1].set_title(f"{model_name}: State {type_S}")
#                         axes[1].set_ylabel(r"$S$", rotation=90)
#                         axes[1].set_xlabel("t")
#                         for elt in group:
#                             sol = elt[0]
#                             N, S = sol.ys.T
#                             timepoints = sol.ts
#                             param_val = elt[1][
#                                 model_vars_and_params_w_indices[k]
#                             ]
#                             axes[0].plot(
#                                 timepoints.tolist(),
#                                 N.tolist(),
#                                 label=rf"{LABELS[k]}={round(param_val, 2)}",
#                             )
#                             axes[1].plot(
#                                 timepoints.tolist(),
#                                 S.tolist(),
#                                 label=rf"{LABELS[k]}={round(param_val, 2)}",
#                             )

#                         axes[0].legend()
#                         axes[1].legend()
#                         # limit setting must come
#                         # after axes.plot()
#                         axes[1].set_xlim(xmin=0)
#                         axes[1].set_ylim(ymin=0)
#                         axes[0].set_xlim(xmin=0)
#                         axes[0].set_ylim(ymin=0)
#                         # plt.show()
#                         # fig = create_plot(model_name, y0s, args, [sols[idx]], style)
#                         pdf.savefig(figure)  # save fig
#                         plt.close(figure)  # close fig after saving
#     else:
#         # create and save a single figure (image format)
#         fig = create_plot(model_name, y0s, args, sols)
#         fig.savefig(file_path, overwrite=overwrite)
#         plt.close(fig)

#     # # plot by groups of variables and parameters
#     # for k, v in model_vars_and_params_w_indices.items():
#     #     # variables or parameters of length
#     #     # one are taken into account below
#     #     if len_model_vars_and_params[k] > 1:
#     #         exclude_index = model_vars_and_params_w_indices[k]
#     #         # NEED to sort groups before
#     #         # it.groupby!!!; remember
#     #         # s_e[1][i] gets the ith entry of
#     #         # y0.tolist() + arg.tolist()
#     #         sorted_group_data = sorted(
#     #             sols_and_entries,
#     #             key=lambda s_e: tuple(
#     #                 s_e[1][i] for i in range(len(s_e[1])) if i != exclude_index
#     #             ),
#     #         )
#     #         groups_for_k = [
#     #             list(group)
#     #             for _, group in it.groupby(
#     #                 sorted_group_data,
#     #                 key=lambda s_e: tuple(
#     #                     s_e[1][i]
#     #                     for i in range(len(s_e[1]))
#     #                     if i != exclude_index
#     #                 ),
#     #             )
#     #         ]
#     #         # plot the group on an individual figure
#     #         # this will only ever plot N or S
#     #         for i, group in enumerate(groups_for_k):
#     #             figure, axes = plt.subplots(nrows=1, ncols=2)
#     #             axes[0].set_title(f"{model_name}: Population Change")
#     #             axes[0].set_ylabel(r"$N$", rotation=90)
#     #             axes[0].set_xlabel("t")
#     #             type_S = "Resources" if model_name == "DFM" else "Wealth"
#     #             axes[1].set_title(f"{model_name}: State {type_S}")
#     #             axes[1].set_ylabel(r"$S$", rotation=90)
#     #             axes[1].set_xlabel("t")
#     #             for elt in group:
#     #                 sol = elt[0]
#     #                 N, S = sol.ys.T
#     #                 timepoints = sol.ts
#     #                 param_val = elt[1][model_vars_and_params_w_indices[k]]
#     #                 axes[0].plot(
#     #                     timepoints.tolist(),
#     #                     N.tolist(),
#     #                     label=rf"{LABELS[k]}={round(param_val, 2)}",
#     #                 )
#     #                 axes[1].plot(
#     #                     timepoints.tolist(),
#     #                     S.tolist(),
#     #                     label=rf"{LABELS[k]}={round(param_val, 2)}",
#     #                 )
#     #             if param_box:
#     #                 figure.subplots_adjust(bottom=0.5)
#     #                 param_list = "; ".join(
#     #                     [
#     #                         f"{LABELS[_]}={', '.join([str(round(e, 2)) for e in config[_]])}"
#     #                         for _ in model_vars_and_params
#     #                         if _ != k
#     #                     ]
#     #                 )
#     #                 figure.text(
#     #                     0.5,  # center
#     #                     0.02,  # near bottom
#     #                     f"Parameters: {param_list}",
#     #                     ha="center",
#     #                     va="top",
#     #                     fontsize=12,
#     #                     bbox=dict(
#     #                         facecolor="white",
#     #                         edgecolor="black",
#     #                         boxstyle="round,pad=0.5",
#     #                     ),
#     #                 )
#     #             axes[0].legend()
#     #             axes[1].legend()
#     #             # limit setting must come
#     #             # after axes.plot()
#     #             axes[1].set_xlim(xmin=0)
#     #             axes[1].set_ylim(ymin=0)
#     #             axes[0].set_xlim(xmin=0)
#     #             axes[0].set_ylim(ymin=0)
#     #             plt.show()


def main(parsed_args: argparse.Namespace) -> None:

    # get configuration file
    config = load_and_validate_config(config_file=parsed_args.config)

    # model specified
    model_selected = config["model"]

    # get model variable and parameter
    # input dictionary
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
    print(sols)

    # possibly plot or save the model results
    if parsed_args.plot:
        # model_vars_and_params = CONFIG_VARS + CONFIG_PARAMS[model_selected]
        # len_model_vars_and_params = {
        #     k: len(config[k]) for k in model_vars_and_params
        # }
        # plot_experiments(
        #     y0s=y0s,
        #     args=input_args,
        #     sols=sols,
        #     model_vars_and_params=model_vars_and_params,
        #     len_model_vars_and_params=len_model_vars_and_params,
        #     model_input=model_input_dict,
        #     save_name=parsed_args.save_name,
        #     style_name=parsed_args.style_name,
        #     separate_plots=parsed_args.separate_plots,
        #     overwrite=parsed_args.overwrite,
        # )
        pass
    if parsed_args.save:
        pass


if __name__ == "__main__":
    # setup and use argument parser for
    # command line arguments
    parser = argparse.ArgumentParser(
        description="Argparser for re-tdwmfc-wittmann. Helps with which model to use, figure to re-create, and whether to save plots."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="The name of the configuration file to use for the model experiment.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the results of the experiment.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Whether to save the results the numerical results of the experiment.",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="",
        help="The name of the file to save outputted results. Required if either --save or --plot is True. Same for output for --save and --plot, just different extensions.",
    )
    parser.add_argument(
        "--style_name",
        type=str,
        default="",
        help="The of the style file without its extension, if desired. This file must be in ./assets/styles/",
    )
    parser.add_argument(
        "--separate_plots",
        action="store_true",
        help="Whether to plot N and S separately.",
    )
    # parser.add_argument(
    #     "--xaxis_value",
    #     type=str,
    #     default="time",
    #     help="Which variable to have plotted on the xaxis. By default, the y-axis for the two subplots will always be N or S.",
    # )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing saved figures or output.",
    )
    # pass the output to main
    parsed_args = parser.parse_args()
    main(parsed_args)
