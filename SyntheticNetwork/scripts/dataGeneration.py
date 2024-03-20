# %%
import os

path = os.path.abspath(__file__)
os.chdir("/".join(path.split(os.sep)[:-3]))
from src.utils.NetworkGenerator import spatial_regional_geometric
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import multiprocessing
import pickle
from src.utils.utils import transfer_to_pytorch_fun
import random
import torch


graph = spatial_regional_geometric(
    node_number=15, k=20, dist=200, seed=None, mapsize=500, method="area"
)

G, pos, Z = graph.generate_graph()
fig, ax = plt.subplots()

edge_list = [
    edge for edge in G.edges() if nx.get_edge_attributes(G, "edge_attr")[edge] > 0
]
nx.draw_networkx(
    G,
    pos,
    node_size=1,
    edge_color="orange",
    ax=ax,
    edgelist=edge_list,
    with_labels=False,
)
im = ax.imshow(Z)
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
plt.colorbar(im, location="right", shrink=0.8)
plt.savefig(
    "SyntheticNetwork/model_results/examplegraph1.png", dpi=300, bbox_inches="tight"
)
plt.show()


# with multiprocessing.Pool() as p:
#     results = p.starmap(graph.generate_graph, [() for _ in range(1000)])

# datalist, pos_list, Z_list = zip(*results)
# datalist = list(datalist)
# with open(f"SyntheticNetwork/data/sampled_nodes.dat", "wb") as f:
#     pickle.dump(datalist, f)

# cut = int(0.7 * len(datalist))
# random.Random(8).shuffle(datalist)
# training_set = datalist[:cut]
# test_set = datalist[cut:]

# num_cores = multiprocessing.cpu_count()

# with multiprocessing.Pool(num_cores) as p:
#     train_set_pytorch = p.map(transfer_to_pytorch_fun, training_set)

# with multiprocessing.Pool(num_cores) as p:
#     test_set_pytorch = p.map(transfer_to_pytorch_fun, test_set)

# print("#### create pytorch graph list####", flush=True)

# torch.save(
#     train_set_pytorch,
#     f"SyntheticNetwork/data/trainig_set.pt",
# )
# torch.save(
#     test_set_pytorch,
#     f"SyntheticNetwork/data/testing_set.pt",
# )

# print("#### data generated successful!!####", flush=True)
