import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
import numpy as np
from src.core.VanilaGCN import Model
from src.core.SpatialGCN import SpatialGCNModel
from src.core.ConvSpatial_single import RSGCNBlock


def plot_a_sample(
    save_fig,
    filename,
    sample_graph,
    show_all_edges,
    threshold=0.5,
    node_size=1,
    pos=None,
    figsize=(6, 5),
    color_bar=True,
    **kwds,
):
    edges = sample_graph.edges()
    if pos is None:
        pos = {
            k: (v["x"][0][1], v["x"][0][0]) for k, v in sample_graph.nodes(data=True)
        }

    if show_all_edges:
        edge_list = list(sample_graph.edges())
    else:
        edge_list = [
            edge
            for edge in edges
            if sample_graph.edges[(edge[0], edge[1])]["edge_attr"] >= threshold
        ]

    edge_color = {
        edge: sample_graph.edges[(edge[0], edge[1])]["edge_attr"] for edge in edge_list
    }
    edgecolor = list(edge_color.values())

    if len(edge_list) > 0:
        fig, ax = plt.subplots(figsize=figsize)
        edges = nx.draw_networkx_edges(
            sample_graph,
            pos,
            edgelist=edge_list,
            ax=ax,
            width=0.5,
            edge_color=edgecolor,
            **kwds,
        )

        nodes = nx.draw_networkx_nodes(sample_graph, pos, node_size=node_size)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        if color_bar:
            plt.colorbar(edges)

        plt.tight_layout()
        plt.gca().set_aspect("equal")

        if save_fig:
            plt.savefig(f"{filename}", bbox_inches="tight", dpi=300)
        plt.show()
        plt.close()

    return sample_graph


def transfer_to_pytorch_fun(G):
    # Create a dictionary of the mappings from company --> node id
    mapping_dict = {x: i for i, x in enumerate(list(G.nodes()))}

    G = nx.relabel_nodes(G, mapping=mapping_dict)
    # Now create a source, target, and edge list for PyTorch geometric graph

    node_feature_list = [nx.get_node_attributes(G, "x")[n][0] for n in G.nodes()]
    node_region_list = [nx.get_node_attributes(G, "x")[n][1] for n in G.nodes()]

    edge_source_list = [edge[0] for edge in G.edges()]
    edge_target_list = [edge[1] for edge in G.edges()]
    edge_weight_list = [
        nx.get_edge_attributes(G, "edge_attr")[edge] for edge in G.edges()
    ]

    # now create full edge lists for pytorch geometric - undirected edges need to be defined in both directions
    full_source_list = edge_source_list + edge_target_list  # full source list
    full_target_list = edge_target_list + edge_source_list  # full target list
    full_weight_list = edge_weight_list + edge_weight_list  # full edge weight list

    # now convert these to torch tensors
    edge_index_tensor = torch.LongTensor(
        np.concatenate([[np.array(full_source_list)], [np.array(full_target_list)]])
    )
    edge_weight_tensor = torch.FloatTensor(np.array(full_weight_list))
    node_feature_tensor = torch.FloatTensor(np.array(node_feature_list))
    node_region_tensor = torch.FloatTensor(np.array(node_region_list))
    data = Data(
        x=node_feature_tensor,
        edge_index=edge_index_tensor,
        edge_attr=edge_weight_tensor,
        region=node_region_tensor,
    )
    return data


def clean_graph_list(results):
    prepared_g = []
    prepared_sample = []
    for result in results:
        if result is not None:
            graph, sample = result[0], result[1]
            if (
                graph is not None
                and len(graph.edges()) < 6000
                and len(graph.nodes()) > 2
                and sum(nx.get_edge_attributes(graph, "edge_attr").values()) > 2
            ):
                prepared_g.append(graph)
                prepared_sample.append(sample)
    return prepared_g, prepared_sample


def return_the_model(model_name, loss_name, valid, device, input_channels):
    if loss_name == "BCE_log" and not valid:
        sigmoid = False
        pos_weight = torch.tensor([7]).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        sigmoid = True
        loss_fn = torch.nn.BCELoss()

    if model_name == "Vanila":
        model = Model(
            input_channels=input_channels, hidden_channels=128, sigmoid=sigmoid
        )
        model.to(device)

    elif model_name == "Spatialgcn":
        model = SpatialGCNModel(
            input_channels=input_channels - 2, hidden_channels=128, sigmoid=sigmoid
        )
        model.to(device)

    elif model_name == "Convspatial":
        model = RSGCNBlock(
            input_channels=input_channels - 2, hidden_channels=128, sigmoid=sigmoid
        )

    else:
        raise ValueError("**Check your model spelling***")
    return model, loss_fn
