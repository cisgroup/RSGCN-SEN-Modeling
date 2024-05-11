# %%
import os
import sys

path = os.path.abspath(__file__)
root_path = "/".join(path.split(os.sep)[:-3])
os.chdir(root_path)
sys.path.append(root_path)
import torch
from src.utils.dataClass import DataPreparation, SamplingGraph
from src.utils.utils import (
    plot_a_sample,
    transfer_to_pytorch_fun,
    return_the_model,
)
import networkx as nx
import numpy as np
import json
from tqdm import tqdm
from src.utils.test_setevaluation import (
    hist_plot,
    bar_plot,
)
import math
from shapely.geometry import Point
from geopandas import GeoSeries
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
import rasterio.plot
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import scienceplots

plt.style.use("science")


def connect_components(rebuild_G, sorted_attr_ed):
    print("The number of connnected component is:")
    print(nx.number_connected_components(rebuild_G))
    while nx.number_connected_components(rebuild_G) != 1:
        minimal_component = min(nx.connected_components(rebuild_G), key=len)
        minimal_graph = rebuild_G.subgraph(minimal_component).copy()
        potential_edges = {
            edge: prob
            for edge, prob in sorted_attr_ed.items()
            if edge[0] in minimal_component or edge[1] in minimal_component
        }

        in_component_edges = []
        for edge, prbability in potential_edges.items():
            if minimal_graph.has_edge(edge[0], edge[1]):
                in_component_edges.append(edge)
        for edge in in_component_edges:
            del potential_edges[edge]

        likely_connect = max(potential_edges, key=potential_edges.get)
        rebuild_G.add_edge(likely_connect[0], likely_connect[1], edge_attr=0.501)

    return rebuild_G


def check_graph(graph):
    if (
        graph is not None
        and len(graph.edges()) < 5000
        and len(graph.nodes()) > 2
        and sum(nx.get_edge_attributes(graph, "edge_attr").values()) > 2
    ):
        return True
    else:
        return False


def read_model(idx):
    with open(f"{dataset}/config/parameters_{idx}.json") as config_file:
        config_params = json.load(config_file)
    model_name = config_params["model_name"]
    loss_name = config_params["loss_name"]

    model, _ = return_the_model(
        model_name=model_name,
        loss_name=loss_name,
        valid=True,
        device=None,
        input_channels=4,
    )
    model.load_state_dict(
        torch.load(
            f"{dataset}/data/trained_model/GNNParameteres_{model_name}.pt",
            map_location="cpu",
        )
    )

    model.to(device)
    return model, model_name


def predict_edges(data, model):
    if model_name == "Convspatial":
        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.region)
    else:
        with torch.no_grad():
            pred = model(data.x, data.edge_index)
    return pred


if __name__ == "__main__":
    dataset = "PowerSystemCase"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_code = {
        "PowerSystemCase": "34",
        "CaliforniaRoad": "06",
    }
    network_data = DataPreparation(
        census_tract=f"{dataset}/data/censustract/census_tract.shp",
        digital_elevation=f"{dataset}/data/DEMData/DEMdata.tif",
        network_data=f"{dataset}/data/network/network.shp",
        dataset=dataset,
        considered_feature=["lon", "lat", "Population density", "Median house value"],
        state_code=state_code[dataset],
        clean_graph=True,
        state_name=None,
    )

    # network_data.network_description()

    points_gdf, G, _ = network_data.construct_node_by_graph()
    pos = {k: (v["x"][0][0], v["x"][0][1]) for k, v in G.nodes(data=True)}

    dataset_type = "Normalized"
    sample_method = "region"
    total_node_feature = {v: nx.get_node_attributes(G, "x")[v][0] for v in G.nodes()}

    total_node_min = np.array(list(total_node_feature.values())).min(0).astype(float)
    total_node_max = np.array(list(total_node_feature.values())).max(0).astype(float)

    network_data = gpd.read_file(f"{dataset}/data/network/network.shp").explode()
    elevation_data = rasterio.open(f"{dataset}/data/DEMData/DEMdata.tif")
    census_tract = gpd.read_file(f"{dataset}/data/censustract/census_tract.shp")

    node = random.sample(G.nodes(), 1)[0]
    node_pos = GeoSeries(
        Point(
            nx.get_node_attributes(G, "x")[node][0][0],
            nx.get_node_attributes(G, "x")[node][0][1],
        )
    )
    nodes_buffer = node_pos.buffer(25000, cap_style=3)
    nodes_buffer_sample = node_pos.buffer(1500, cap_style=3)

    fig, ax = plt.subplots()
    network_data.plot(ax=ax, alpha=0.7, color="black")
    # nodes_buffer.plot(ax=ax, alpha=0.3, color="blue")
    # nodes_buffer_sample.plot(ax=ax, alpha=0.5, color="red")
    # census_tract.plot(ax=ax, alpha=0.5)
    rasterio.plot.show(elevation_data, ax=ax, vmin=-3, vmax=100)

    # plt.savefig(f"{dataset}/model_results/sample_visulization.png", bbox_inches="tight")
    plt.show()

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()

    Total_Graph = SamplingGraph(
        graph=G,
        planar_graph=None,
        sample_method=sample_method,
        dataset_type=dataset_type,
        window_size=25000,
        total_node_min=total_node_min,
        total_node_max=total_node_max,
    )

    # idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    idx = 1
    model, model_name = read_model(idx)
    overall_edges = []

    for node in tqdm(list(G.nodes())):
        final_graph, _ = Total_Graph.sample_network(points_df=points_gdf, node=node)
        if check_graph(final_graph):
            subgraph = transfer_to_pytorch_fun(final_graph).to(device)
            pred_edges = predict_edges(data=subgraph, model=model)
            edge_dict = {e: k for e, k in zip(final_graph.edges(), pred_edges.tolist())}
            del pred_edges
            torch.cuda.empty_cache()
            overall_edges.append(edge_dict)

        else:
            continue

    final = {
        k: [d.get(k, np.nan) for d in overall_edges]
        for k in set().union(*overall_edges)
    }

    edge_attr_ed = {k: np.nanmedian(np.array(value)) for k, value in final.items()}
    sorted_attr_ed = {}
    sorted_keys = [tuple(sorted(edge)) for edge in edge_attr_ed.keys()]
    sorted_values = [prob for prob in edge_attr_ed.values()]

    for key in sorted_keys:
        hits = (i for i, value in enumerate(sorted_keys) if value == key)
        sorted_attr_ed[key] = max([sorted_values[t] for t in hits])

    remove_edges = [edge for edge, item in sorted_attr_ed.items() if item < 0.5]
    rebuild_G = nx.from_edgelist(list(sorted_attr_ed.keys()))
    nx.set_edge_attributes(rebuild_G, sorted_attr_ed, "edge_attr")
    edge_longer = [e for e in G.edges if not rebuild_G.has_edge(e[0], e[1])]

    rebuild_G.remove_edges_from(remove_edges)

    rebuild_G.add_edges_from(edge_longer, edge_attr=0.501)

    rebuild_G = connect_components(rebuild_G=rebuild_G, sorted_attr_ed=sorted_attr_ed)

    plot_a_sample(
        save_fig=True,
        filename=f"{dataset}/model_results/Original Map.png",
        sample_graph=G,
        show_all_edges=True,
        threshold=0.5,
        pos=pos,
        node_size=0,
        # figsize=(6.5, 12.5),
        color_bar=False,
    )

    plot_a_sample(
        save_fig=True,
        filename=f"{dataset}/model_results/Predicted Map{model_name}.png",
        sample_graph=rebuild_G,
        show_all_edges=True,
        threshold=0.5,
        pos=pos,
        node_size=0,
        # figsize=(8.5, 12.5),
    )

    data_distance = {edge: math.dist(pos[edge[0]], pos[edge[1]]) for edge in G.edges()}

    rebuild_distance = {
        edge: math.dist(pos[edge[0]], pos[edge[1]]) for edge in rebuild_G.edges()
    }

    hist_plot(
        dataset,
        list(data_distance.values()),
        list(rebuild_distance.values()),
        "Edge_length_Network",
        model_name,
    )

    data_node = [G.degree(weight="edge_attr")[n] for n in G.nodes()]

    predict_node = [rebuild_G.degree()[n] for n in rebuild_G.nodes()]

    bar_plot(dataset, data_node, predict_node, "Node degree_Network", model_name)

    predict_edges = [1 if v >= 0.5 else 0 for k, v in edge_attr_ed.items()] + [
        1 for e in edge_longer
    ]
    ground_truth = [
        1 if G.has_edge(k[0], k[1]) else 0 for k, v in edge_attr_ed.items()
    ] + [1 for e in edge_longer]

    cm = confusion_matrix(ground_truth, predict_edges)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["non-edges", "edges"]
    )
    disp.plot(cmap=plt.cm.Blues)
    # disp.ax_.get_images()[0].set_clim(0, 500)
    disp.im_.set_clim(0, 700)
    disp.im_.colorbar.remove()
    plt.savefig(f"{dataset}/model_results/ConfusionMatrix.png")

    plt.show()

    print(f1_score(predict_edges, ground_truth))
