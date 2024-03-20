# %%
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import os

path = os.path.abspath(__file__)
os.chdir("/".join(path.split(os.sep)[:-3]))
from pathlib import Path
from src.utils.utils import plot_a_sample

import scienceplots
import random
import numpy as np

plt.style.use("science")


def hist_plot(data, title):
    _, bin_edges = np.histogram(data)
    plt.hist(data, bins=bin_edges, edgecolor="black", alpha=0.5)

    plt.savefig(f"{Case_study}/model_results/{title}distribution.png")
    plt.show()


def bar_plot(data, title):
    x_data, counts_data = np.unique(data, return_counts=True)

    plt.bar(x_data, counts_data, alpha=0.5)
    plt.savefig(f"{Case_study}/model_results/{title}distribution.png")
    plt.show()


if __name__ == "__main__":
    Case_study = "SyntheticNetwork"

    with open(f"{Case_study}/data/sampled_nodes.dat", "rb") as f:
        graphlist_sample = pickle.load(f)

    idx = random.randint(0, len(graphlist_sample) - 1)
    sample_graph_original = graphlist_sample[idx]

    _ = plot_a_sample(
        save_fig=False,
        sample_graph=sample_graph_original,
        show_all_edges=False,
        filename=f"{Case_study}/model_results/Roadsampled_example.png",
    )

    edge_distance, node_degree, num_edge, num_node = [], [], [], []

    for graph in graphlist_sample:
        num_node += [len(graph.nodes())]
        num_edge += [len(graph.edges())]
        data_edge = [
            edge
            for edge in graph.edges()
            if nx.get_edge_attributes(graph, "edge_attr")[edge] > 0.5
        ]

        data_distance = [
            nx.get_edge_attributes(graph, "weight")[edge] for edge in data_edge
        ]

        data_node = [graph.degree(weight="edge_attr")[n] for n in graph.nodes()]

        edge_distance += data_distance
        node_degree += data_node

    # hist_plot(edge_distance, "Edge length")
    # bar_plot(node_degree, "Node degree")
