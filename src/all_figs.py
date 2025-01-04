"""
A script to create all figure using the
exp_setup.py program.
"""

# %% LIBRARIES USED

import subprocess

# %% EXECUTE FUNCTION


def run_experiments():
    """
    Run experiments for specified figures
    with subprocess calls in a loop.
    """
    figures = [1, 2, 3, 8, 9, 10, 11]

    for fig in figures:
        fig_val = f"0{str(fig)}" if fig < 10 else str(fig)
        config_path = f"../config/fig_{fig_val}.toml"
        style_path = "../assets/styles/general_AF.mplstyle"
        plot_output_path = "../assets/figures"
        save_output_path = "../assets/experiments"

        # plot command
        plot_command = [
            "python3",
            "exp_setup.py",
            "--config",
            config_path,
            "--plot",
            "--style_path",
            style_path,
            "--output_path",
            plot_output_path,
        ]

        # save command
        save_command = [
            "python3",
            "exp_setup.py",
            "--config",
            config_path,
            "--save",
            "--output_path",
            save_output_path,
        ]

        # run plot command
        print(f"Running plot command for fig_{fig_val}")
        subprocess.run(plot_command, check=True)

        # run save command
        print(f"Running save command for fig_{fig_val}")
        subprocess.run(save_command, check=True)


run_experiments()

# %%
