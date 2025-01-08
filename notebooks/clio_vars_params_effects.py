"""
A script capturing the relationships between
different cliodynamics variables and effects.
Some of the values were taken from Table 01
of page 4 in (The Demographic-Wealth
model for cliodynamics), 2024, by Wittmann
and Kuehn.
"""

import matplotlib.pyplot as plt
import networkx as nx

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
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="lightblue",
    edge_color=edge_colors,
    arrowsize=20,
    font_size=10,
)
plt.title("Directed Graph of Variables and Relationships", fontsize=14)
plt.show()


additional_variables = [
    "Infrastructure Quality",
    "Technology Level",
    "Healthcare",
    "Trade Wealth",
    "Environmental Resources",
    "Public Discontent",
    "Education Level",
    "Corruption Level",
    "Elite Fragmentation",
    "Cultural Unity",
]
