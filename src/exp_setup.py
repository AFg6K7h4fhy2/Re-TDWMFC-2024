# """
# Sets up an experiment for use with the
# Demographic Fiscal Model (DFM) or Demographic
# Wealth Model (DWM) by reading in variable and
# parameter values from configuration files.
# The purpose of this file is to permit easy
# visualization capabilities for different
# combinations of parameters. These files
# are meant to be run from within `./src`.

# To run w/ normal plots:
# python3 exp_setup.py --config "fig_01.toml"
# """

# import argparse
# import datetime as dt
# import itertools as it
# import pathlib
# import time
# from collections.abc import Sequence
# from typing import Any

# import diffrax
# import jax
# import jax.numpy as jnp
# import matplotlib.pyplot as plt
# import toml
# from matplotlib.backends.backend_pdf import PdfPages

# from DFM import DFM
# from DWM import DWM

# # parameters for model running that ought
# # never to have multiple values defined
# # for them in a configuration file
# CONFIG_SPECS = ["t0", "t1", "dt0"]

# # the variables (population and state
# # resources, in case of DFM, or wealth, in
# # case of DWM)
# CONFIG_VARS = ["init_N", "init_S"]

# # currently supported models
# SUPPORTED_MODELS = ["DFM", "DWM"]
# MODELS = {"DFM": DFM, "DWM": DWM}

# # additional models can be added here
# CONFIG_PARAMS = {
#     # the parameters that the DFM model needs
#     # and or accepts; the order must match
#     # that in the DFM function i.e.
#     # r, init_rho, beta, init_k, c, init_s = args
#     "DFM": [
#         "r",  # population growth rate
#         "init_rho",  # taxation rate
#         "beta",  # expenditure rate
#         "init_k",  # initial carrying capacity
#         "c",  # max_k - init_k
#         "init_s",  # initial state resources
#     ],
#     # the parameters that the DFM model needs
#     # and or accepts; the order must match
#     # that in the DWM function i.e.
#     # r, beta, alpha, d, g, c, init_k = args
#     "DWM": [
#         "r",  # population growth rate
#         "beta",  # expenditure rate
#         "alpha",  # ?
#         "d",  # strength of negative feedback from S to N
#         "g",  # tax rate times the fraction of surplus gained through investing/expanding
#         "c",  # max_k - init_k
#         "init_k",  # initial carrying capacity
#     ],
# }

# # the LaTeX labels for different variables
# # and parameters used across the DWM and DFM
# # models
# LABELS = {
#     "init_N": r"$N_0$",
#     "init_S": r"$S_0$",
#     "init_rho": r"$\rho_0$",
#     "init_s": r"$s_0$",
#     "init_k": r"$k_0$",
#     "max_k": r"$k_{\text{max}}$",
#     "c": r"$c$",
#     "r": r"$r$",
#     "beta": r"$\beta$",
#     "alpha": r"$\alpha$",
#     "g": r"$g$",
#     "d": r"$d$",
# }


# def check_values_interval(
#     values: list[int] | list[float],
#     min_value: int | float,
#     max_value: int | float,
# ) -> bool:
#     """
#     Checks whether all numerical elements of
#     a list are within specified bounds.

#     Parameters
#     ----------
#     values : list[int] | list[float]
#         Variables or parameters.
#     min_value : int | float
#         The lower bound (inclusive).
#     max_value : int | float
#         The upper bound (inclusive).

#     Returns
#     -------
#     bool
#         Whether all values are within the
#         specified bounds.
#     """
#     # make sure all elements are int or float
#     if not all(isinstance(value, (int, float)) for value in values):
#         raise TypeError(
#             f"All values must be either int or float; got {values}."
#         )
#     # make sure all elements are in bounds
#     if all(min_value <= value <= max_value for value in values):
#         return True
#     else:
#         raise ValueError(
#             f"All values must be between {min_value} and {max_value}."
#         )


# def ensure_listlike(x: Any) -> Sequence[Any]:
#     """
#     Ensures that an element is listlike,
#     i.e. a Sequence.

#     Parameters
#     ----------
#     x : Any
#         An object intended to be listlike.

#     Returns
#     -------
#     Sequence[Any]
#         The object if already listlike or
#         a list containing the object.
#     """
#     return x if isinstance(x, Sequence) else [x]


# def load_and_validate_config(
#     config_file: str,
# ) -> dict[str, str | float | int | list[int] | list[float]]:
#     """
#     Extract content specified in a TOML
#     configuration file.

#     Parameters
#     ----------
#     config_file : str
#         The name of the config file.

#     Returns
#     -------
#     dict[str, str | float | int | list[int] | list[float]]
#         A dictionary of model specifications,
#         parameters, and variables. The
#         following parameters and variables
#         are permitted to be lists: init_N,
#         init_S, init_rho, init_s, init_k,
#         max_k, c, r, beta.
#     """
#     # the established config location;
#     # assumed one is running code in ./src
#     base_path = pathlib.Path("../config")
#     config_path = base_path / config_file
#     # confirm the config file exists
#     if not config_path.is_file():
#         raise FileNotFoundError(f"Config file not found: {config_path}")
#     # attempt loading the toml config file
#     try:
#         config = toml.load(config_path)
#     except Exception as e:
#         raise Exception(f"Error while loading TOML: {e}")
#     # ensure that all loaded configuration
#     # entries are proper
#     loaded_entries = list(config.keys())
#     if "model" not in loaded_entries:
#         raise ValueError(
#             'There is currently not "model" key in the loaded configuration elements.'
#         )
#     model_specified = config["model"]
#     if model_specified not in SUPPORTED_MODELS:
#         raise ValueError(
#             f"The specified model ({model_specified}) is not in the supported models: {SUPPORTED_MODELS}."
#         )
#     missing_model_vals = [
#         val
#         for val in CONFIG_SPECS + CONFIG_VARS + CONFIG_PARAMS[model_specified]
#         if val not in loaded_entries
#     ]
#     if missing_model_vals:
#         raise ValueError(
#             f"The following values ({missing_model_vals}) are missing for the {model_specified} model."
#         )
#     # ensure all config entries are listlike
#     vars_to_make_listlike = CONFIG_VARS + CONFIG_PARAMS[model_specified]
#     for k, v in config.items():
#         if not isinstance(v, list) and k in vars_to_make_listlike:
#             config[k] = ensure_listlike(v)
#     # ensure variables and parameters are
#     # within the correct intervals
#     # check_values_interval(
#     #     values=config["init_N"], min_value=0.01, max_value=5.0
#     # )
#     # check_values_interval(
#     #     values=config["init_S"], min_value=0.0, max_value=5.0
#     # )
#     # check_values_interval(values=config["init_rho"], min_value=1, max_value=4)
#     # check_values_interval(values=config["init_s"], min_value=1, max_value=30)
#     # check_values_interval(values=config["init_k"], min_value=1, max_value=10)
#     # check_values_interval(values=config["max_k"], min_value=1, max_value=10)
#     # check_values_interval(values=config["c"], min_value=1, max_value=10)
#     # check_values_interval(values=config["r"], min_value=0.01, max_value=0.90)
#     # check_values_interval(
#     #     values=config["beta"], min_value=0.00, max_value=0.90
#     # )
#     # max_init_k = max(config["init_k"])
#     # min_max_k = min(config["max_k"])
#     # if max_init_k >= min_max_k:
#     #     raise ValueError(
#     #         f"Minimum max carry capacity (got {min_max_k}) must be greater than initial carry capacity (got {max_init_k})."
#     #     )
#     return config


# def get_y0s(
#     init_N: list[float] | list[int], init_S: list[float] | list[int]
# ) -> list[jax.Array]:
#     """
#     TODO:
#     """
#     y0s = [jnp.array(pair) for pair in list(it.product(init_N, init_S))]
#     return y0s


# def get_args(
#     model_input: dict[str, list[int] | list[float]]
# ) -> list[jax.Array]:
#     """
#     TODO:
#     """
#     args = [
#         jnp.array(group)
#         for group in list(
#             it.product(
#                 *list(model_input.values()),
#             )
#         )
#     ]
#     return args


# def run_clio_model(
#     t0: int,
#     t1: int,
#     dt0: int,
#     model_name: str,
#     y0s: list[jax.Array],
#     args: list[jax.Array],
# ) -> list[jax.Array]:
#     """
#     Run a single cliodynamics model.
#     """
#     saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, t1 - t0))
#     solver = diffrax.Tsit5()
#     term = diffrax.ODETerm(MODELS[model_name])
#     sols = [
#         diffrax.diffeqsolve(
#             term, solver, t0, t1, dt0, y0, args=arg, saveat=saveat
#         )
#         for y0 in y0s
#         for arg in args
#     ]
#     return sols


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
#     if param_box:
#         figure.subplots_adjust(bottom=0.5)
#         param_list = "; ".join(
#             [
#                 f"{LABELS[_]}={', '.join([str(round(e, 2)) for e in config[_]])}"
#                 for _ in model_vars_and_params
#                 if _ != k
#             ]
#         )
#         figure.text(
#             0.5,  # center
#             0.02,  # near bottom
#             f"Parameters: {param_list}",
#             ha="center",
#             va="top",
#             fontsize=12,
#             bbox=dict(
#                 facecolor="white",
#                 edgecolor="black",
#                 boxstyle="round,pad=0.5",
#             ),
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


# def plot_and_save(
#     model_name: str,
#     config: dict[str, str | float | int | list[int] | list[float]],
#     y0s: list[jax.Array],
#     args: list[jax.Array],
#     sols: list[jax.Array],
#     model_input: dict[str, list[float] | list[int]],
#     save_name: str,
#     overwrite: bool = False,
#     param_box: bool = False,
#     style: str = None,
# ) -> None:
#     """
#     Plots and saves executed cliodynamics
#     model results. If there are multiple
#     plots, they're saved as a PDF; otherwise,
#     they're saved as an image. The file name
#     included the model name, the configuration
#     file, and the date as (ISO 8601). The PDF
#     or image file metadata contains the
#     Python code and toml file contents for the
#     images produced therein. Overwrite defaults
#     to False. Parameter box defaults to False.

#     Parameters
#     ----------
#     model_name : str
#         The name of the model used. Usually is
#         either DFM or DWM.
#     config : dict[str, str | float | int | list[int] | list[float]]
#         A dictionary of model specifications,
#         parameters, and variables. The
#         following parameters and variables
#         are permitted to be lists: init_N,
#         init_S, init_rho, init_s, init_k,
#         max_k, c, r, beta.
#     y0s : list[jax.Array]
#         The initial variable values for
#         the model.
#     args : list[jax.Array]
#         Different combinations of parameters
#         for the specified model.
#     sols : list[jax.Array]
#         A list of ODE solutions corresponding
#         to different combinations of variables
#         and parameters of the specified model.
#     model_input : dict[str, list[float] | list[int]]
#         Parameters and their values for the
#         specified model.
#     save_name : str
#         What to name the output file.
#     overwrite : bool
#         Whether to overwrite a saved output
#         file. Defaults to False.
#     param_box : bool
#         Whether to use a parameter box showing
#         the values of parameters used in the
#         model.
#     style : str
#         The name of the style file to use for
#         plotting. Defaults to None, i.e.
#         matplotlib's default style.

#     Returns
#     -------
#     None
#         A PDF or image of the figures produced.
#     """
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
#     # plot as an image if there is only a
#     # single plot (all vars and params len 1
#     # or just one that is not len 1),
#     # otherwise plot as a pdf
#     plot_as_img = (
#         sum(
#             [
#                 0 if len(elt) == 1 else 1
#                 for elt in len_model_vars_and_params.values()
#             ]
#         )
#         <= 1
#     )
#     # by default, this function looks for
#     # ../assets/figures as a directory...
#     current_date = dt.datetime.now().isoformat()
#     output_dir = "../assets/figures"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         raise FileNotFoundError(f"The directory {output_dir} does not exist.")
#     file_name = (
#         f"{save_name}_{current_date}.png"
#         if plot_as_img
#         else f"{save_name}_{current_date}.pdf"
#     )
#     file_path = os.path.join(output_dir, file_name)

#     # associate the correctly ordered variables
#     # and parameters with indices
#     model_vars_and_params = CONFIG_VARS + CONFIG_PARAMS[model_name]
#     len_model_vars_and_params = {
#         k: len(config[k]) for k in model_vars_and_params
#     }
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
#                         if param_box:
#                             figure.subplots_adjust(bottom=0.5)
#                             param_list = "; ".join(
#                                 [
#                                     f"{LABELS[_]}={', '.join([str(round(e, 2)) for e in config[_]])}"
#                                     for _ in model_vars_and_params
#                                     if _ != k
#                                 ]
#                             )
#                             figure.text(
#                                 0.5,  # center
#                                 0.02,  # near bottom
#                                 f"Parameters: {param_list}",
#                                 ha="center",
#                                 va="top",
#                                 fontsize=12,
#                                 bbox=dict(
#                                     facecolor="white",
#                                     edgecolor="black",
#                                     boxstyle="round,pad=0.5",
#                                 ),
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


# def main(parsed_args: argparse.Namespace) -> None:

#     # get configuration file
#     config = load_and_validate_config(config_file=parsed_args.config)

#     # model specified
#     model_selected = config["model"]

#     # get model variable and parameter
#     # input dictionary
#     model_input_dict = {k: config[k] for k in CONFIG_PARAMS[model_selected]}

#     # gets y0s and args for model
#     y0s = get_y0s(init_N=config["init_N"], init_S=config["init_S"])
#     input_args = get_args(model_input=model_input_dict)

#     # run model combinations, getting the
#     # time as well
#     start = time.time()
#     sols = run_clio_model(
#         t0=config["t0"],
#         t1=config["t1"],
#         dt0=config["dt0"],
#         model_name=model_selected,
#         y0s=y0s,
#         args=input_args,
#     )
#     elapsed = time.time() - start
#     print(
#         f"Experiments Using {model_selected} Ran In:\n{round(elapsed, 5)} Seconds.\n"
#     )

#     # plot output
#     plot_and_save(
#         model_name=model_selected,
#         config=config,
#         y0s=y0s,
#         args=input_args,
#         sols=sols,
#         model_input=model_input_dict,
#         style="general_AF",
#     )


# if __name__ == "__main__":
#     # setup and use argument parser
#     parser = argparse.ArgumentParser(
#         description="Argparser for re-tdwmfc-wittmann. Helps with which model to use, figure to re-create, and whether to save plots."
#     )
#     parser.add_argument(
#         "--config",
#         type=str,
#         required=True,
#         help="The name of the configuration file to use.",
#     )
#     # TODO: plot (just show output)
#     # TODO: store sols as json
#     # TODO: style
#     # TODO: use param box
#     # TODO:

#     # parser.add_argument(
#     #     "--style",
#     #     type=str,
#     #     default="default",
#     #     help="(optional) The name of the style file to use.",
#     # )
#     # parser.add_argument(
#     #     "--save_as_pdf",
#     #     action="store_true",
#     #     help="(optional) Whether to save plots that were generated as PDFs.",
#     # )
#     # parser.add_argument(
#     #     "--save_as_img",
#     #     action="store_true",
#     #     help="(optional) Whether to save plots that were generated as images.",
#     # )
#     parser.add_argument(
#         "--save_name",
#         type=str,
#         default="default_name",
#         help="The save name for figure(s).",
#     )
#     parser.add_argument(
#         "--plot_on_x",
#         type=str,
#         default="time",
#         help="The name of the variable to plot on the x-axis.",
#     )
#     # parser.add_argument(
#     #     "--save_path",
#     #     type=str,
#     #     default="../assets/images",
#     #     help="Where to save figures.",
#     # )
#     # parser.add_argument(
#     #     "--overwrite",
#     #     action="store_true",
#     #     help="Whether to overwrite existing saved figures.",
#     # )
#     # parser.add_argument(
#     #     "--param_box",
#     #     action="store_true",
#     #     help="Whether to have the parameters and variables in a box in the figures.",
#     # )
#     parsed_args = parser.parse_args()
#     # pass args to main and execute model
#     main(parsed_args)
