# %%
import pickle
import os
import sys


path = os.path.abspath(__file__)
root_path = "/".join(path.split(os.sep)[:-3])
os.chdir(root_path)
sys.path.append(root_path)
import random
import multiprocessing
from functools import partial
import torch
from src.utils.dataClass import DataPreparation, SamplingGraph
from src.utils.utils import plot_a_sample, transfer_to_pytorch_fun, clean_graph_list
import networkx as nx
import numpy as np


torch.multiprocessing.set_sharing_strategy("file_system")

if __name__ == "__main__":
    dataset = "PowerSystemCase"

    state_code = {
        "PowerSystemCase": "34",
        "CaliforniaRoad": "06",
    }

    power_data = DataPreparation(
        census_tract=f"{dataset}/data/censustract/census_tract.shp",
        digital_elevation=f"{dataset}/data/DEMData/DEMdata.tif",
        network_data=f"{dataset}/data/network/network.shp",
        dataset=dataset,
        considered_feature=["lon", "lat", "Population density", "Median house value"],
        state_code=state_code[dataset],
        clean_graph=True,
        state_name=None,
    )

    # power_data.network_description()

    points_gdf, G, _ = power_data.construct_node_by_graph()
    pos = {k: (v["x"][0][0], v["x"][0][1]) for k, v in G.nodes(data=True)}
    dataset_type = "Normalized"
    sample_method = "region"
    total_node_feature = {v: nx.get_node_attributes(G, "x")[v][0] for v in G.nodes()}
    total_node_min = np.array(list(total_node_feature.values())).min(0).astype(float)
    total_node_max = np.array(list(total_node_feature.values())).max(0).astype(float)

    Total_Graph = SamplingGraph(
        graph=G,
        planar_graph=None,
        sample_method=sample_method,
        dataset_type=dataset_type,
        window_size=25000,
        total_node_min=total_node_min,
        total_node_max=total_node_max,
    )

    num_cores = multiprocessing.cpu_count()
    sampled_nodes = list(G.nodes())

    with multiprocessing.Pool(num_cores) as p:
        results = p.map(
            partial(Total_Graph.sample_network, points_df=points_gdf), sampled_nodes
        )

    datalist, sample_list = clean_graph_list(results)

    print("#### create graph list####", flush=True)

    with open(
        f"{dataset}/data/sampled_nodes{dataset_type}{sample_method}.dat", "wb"
    ) as f:
        pickle.dump(datalist, f)

    with open(
        f"{dataset}/data/sampled_nodes{dataset_type}{sample_method}Samples.dat", "wb"
    ) as f:
        pickle.dump(sample_list, f)

    cut = int(0.7 * len(datalist))
    random.Random(8).shuffle(datalist)
    training_set = datalist[:cut]
    test_set = datalist[cut:]
    with multiprocessing.Pool(num_cores) as p:
        train_set_pytorch = p.map(transfer_to_pytorch_fun, training_set)

    with multiprocessing.Pool(num_cores) as p:
        test_set_pytorch = p.map(transfer_to_pytorch_fun, test_set)

    print("#### create pytorch graph list####", flush=True)

    torch.save(
        train_set_pytorch,
        f"{dataset}/data/trainig_set{dataset_type}{sample_method}.pt",
    )
    torch.save(
        test_set_pytorch,
        f"{dataset}/data/testing_set{dataset_type}{sample_method}.pt",
    )

    print("#### data generated successful!!####", flush=True)

    plot_a_sample(
        save_fig=False,
        filename=f"{dataset}/model_results/Original Map.png",
        sample_graph=training_set[random.randint(0, 100)],
        show_all_edges=False,
        threshold=0.5,
        pos=pos,
        node_size=0,
    )
