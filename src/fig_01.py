"""
Recreates figure one, "Dynamics of model",
pp. 06/21, of book Demographic Fiscal Model
(DFM) from Historical Dynamics
(Turchin, 2003, pp.121) as stated in the
2024 paper (The Demographic-Wealth model for
cliodynamics) by Wittmann and Kuehn.
"""

# %% IMPORTS

import pathlib

import matplotlib.pyplot as plt

# %% LOAD STYLE SHEET IF AVAILABLE

base_style_path = pathlib.Path("../assets/styles")
style = "fig_replication"
style_path = base_style_path / (style + ".mplstyle")
if style_path.exists():
    plt.style.use(str(style_path))

# %% DEFINE PARAMETERS (see fig_01.toml)

t0 = 0
t1 = 500
dt0 = 1
init_S = 0.0
init_N = 0.5
init_p = 1
init_s = 10
init_k = 1
max_k = 4
c = 3
r = 0.02
beta = [0.0, 0.1, 0.25, 0.4]

#
