# Variables And Parameters For Cliodynamics Models

The following table covers possible variables and parameters, along with their possible interaction, from page 4 of [^dwm].

[^dwm]: Wittmann, Lukas, and Christian Kuehn. "The Demographic-Wealth model for cliodynamics." Plos one 19, no. 4 (2024): e0298318.

> Table 1 provides a possible, yet highly debatable, attempt to identify some time-dependent macro state variables (leftmost column) and how they might depend upon certain observed processes/events such as population growth, carrying capacity, high state costs, rebellions/wars, changes in political power, etc. Signs indicate whether the event/process might positively (+) or negatively (−) change the dynamics of the macro-variables. It is evident that there could be many more possible variables, and already the positive/negative influence modeling is difficult. This does not even address the matter of possible functional form relationships to express the model precisely.

| Macro-Variable       | Population Growth | Carrying Capacity | High State Costs | Rebellions & Wars | Power Change | ... |
|-----------------------|-------------------|-------------------|------------------|-------------------|--------------|-----|
| Population Size       | +                 | -                 | +                | -                 | +/-          | ... |
| State Wealth          | +                 | -                 | -                | -                 | +/-          | ... |
| Taxation Rate         | +                 | -                 | +                | +                 | +/-          | ... |
| Nouveaux Riches       | +                 | +/-               | +/-              | -                 | +/-          | ... |
| Nouveaux Pauvres      | +                 | +                 | -                | -                 | +/-          | ... |
| Army Size             | +                 | -                 | -                | +                 | +/-          | ... |
| Social Balance        | -                 | -                 | +                | -                 | +/-          | ... |
| ...                   | ...               | ...               | ...              | ...               | ...          | ... |

```python
rows = [
    "Population Size",
    "State Wealth",
    "Taxation Rate",
    "Nouveaux Riches",
    "Nouveaux Pauvres",
    "Army Size",
    "Social Balance",
]

columns = [
    "Population Growth",
    "Carrying Capacity",
    "High State Costs",
    "Rebellions & Wars",
    "Power Change",
]

import networkx as nx
import matplotlib.pyplot as plt

variables = [
    "Population Size",
    "State Wealth",
    "Taxation Rate",
    "Nouveaux Riches",
    "Nouveaux Pauvres",
    "Army Size",
    "Social Balance",
]

relationships = {
    "Population Growth": ["+", "+", "+", "+", "+", "+", "-"],
    "Carrying Capacity": ["-", "-", "-", "+/-", "+", "-", "-"],
    "High State Costs": ["+", "-", "+", "+/-", "-", "-", "+"],
    "Rebellions & Wars": ["-", "-", "+", "-", "-", "+", "-"],
    "Power Change": ["+/-", "+/-", "+/-", "+/-", "+/-", "+/-", "+/-"],
}

G = nx.DiGraph()

for var in variables:
    G.add_node(var)

for target, effects in relationships.items():
    for i, effect in enumerate(effects):
        if effect == "+":
            G.add_edge(variables[i], target, color="green")
        elif effect == "-":
            G.add_edge(variables[i], target, color="red")
        elif effect == "+/-":
            G.add_edge(variables[i], target, color="blue")

edge_colors = [G[u][v]["color"] for u, v in G.edges()]

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)  # Set layout
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color=edge_colors, arrowsize=20, font_size=10)
plt.title("Directed Graph of Variables and Relationships", fontsize=14)
plt.show()

```
