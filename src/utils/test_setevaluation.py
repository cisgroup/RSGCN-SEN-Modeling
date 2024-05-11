import os
import numpy as np

path = os.path.abspath(__file__)
os.chdir("/".join(path.split(os.sep)[:-3]))
import random
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

from src.utils.utils import transfer_to_pytorch_fun, plot_a_sample
import json
import networkx as nx
from scipy import stats


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict_cm(dataset, model_name, randomgraph, model, threshold, device):
    data = transfer_to_pytorch_fun(randomgraph).to(device)
    if model_name == "Convspatial":
        pred = model(data.x, data.edge_index, data.region)
    else:
        pred = model(data.x, data.edge_index)

    pred_edge = torch.where(pred < threshold, 0, 1)
    real_edge = data.edge_attr

    pred_cpu = pred_edge.tolist()
    pred_float_cpu = pred.tolist()

    cm = confusion_matrix(real_edge.tolist(), pred_cpu)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig(
        f"{dataset}/model_results/confusionmatrix_power_{model_name}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()
    plt.close()

    pred_edge_feature = {e: k for e, k in zip(randomgraph.edges(), pred_float_cpu)}
    nx.set_edge_attributes(randomgraph, pred_edge_feature, "edge_attr")

    return randomgraph


def sample_validation(
    dataset, model_name, model, threshold, test_set, device, seed=None
):
    random.seed(seed)
    randomgraph = random.choice(test_set).copy()

    plot_a_sample(
        save_fig=True,
        filename=f"{dataset}/model_results/Example.png",
        show_all_edges=False,
        sample_graph=randomgraph,
    )

    model.eval()
    network_G = predict_cm(dataset, model_name, randomgraph, model, threshold, device)
    plot_a_sample(
        filename=f"{dataset}/model_results/{model_name}Prediction.png",
        sample_graph=network_G,
        show_all_edges=False,
        save_fig=True,
        threshold=threshold,
        edge_vmin=0.5,
        edge_vmax=1.0,
    )


def hist_plot(dataset, data, prediction, title, model_name):
    data_density, bin_edges = np.histogram(data, bins=12, density=True)
    plt.hist(data, bins=bin_edges, edgecolor="black", label="data", alpha=0.5)

    if prediction is not None:
        prediction_density, _ = np.histogram(prediction, bins=bin_edges, density=True)
        data_entropy = data_density[prediction_density > 0]
        prediction_entropy = prediction_density[prediction_density > 0]

        rmse_error = stats.entropy(data_entropy, prediction_entropy)

        plt.hist(
            prediction, bins=bin_edges, edgecolor="black", label="prediction", alpha=0.5
        )
        plt.xlabel("Edge Length (m)")
        plt.ylabel("Number")

        with open(f"{dataset}/model_results/output.txt", "a") as f:
            f.write(
                f"\n***: **{dataset}: the distance of {title} using model {model_name} is {rmse_error}!****\n"
            )

    plt.legend(loc="upper right")
    plt.savefig(f"{dataset}/model_results/{model_name}{title}distribution.png")
    plt.show()
    plt.close()


def bar_plot(dataset, data, prediction, title, model_name):
    x_data, counts_data = np.unique(data, return_counts=True)
    x_prediction, counts_prediction = np.unique(prediction, return_counts=True)
    data_dict = {x: y for x, y in zip(x_data, counts_data)}
    prediction_dict = {x: y for x, y in zip(x_prediction, counts_prediction)}

    counts_data_post = np.empty(max(len(x_data), len(x_prediction)))
    counts_prediction_post = np.empty(max(len(x_data), len(x_prediction)))
    for i in range(max(len(x_data), len(x_prediction))):
        if i in data_dict.keys():
            counts_data_post[i] = data_dict[i]
        else:
            counts_data_post[i] = 0

        if i in prediction_dict.keys():
            counts_prediction_post[i] = prediction_dict[i]
        else:
            counts_prediction_post[i] = 0

    plt.bar(
        range(max(len(x_data), len(x_prediction))),
        counts_data_post,
        label="data",
        alpha=0.5,
    )
    plt.bar(
        range(max(len(x_data), len(x_prediction))),
        counts_prediction_post,
        label="prediction",
        alpha=0.5,
    )
    plt.legend(loc="upper right")
    plt.xlabel("Node degree")
    plt.ylabel

    plt.savefig(f"{dataset}/model_results/{model_name}{title}distribution.png")
    plt.show()
    plt.close()

    counts_data_entropy = counts_data_post[counts_prediction_post > 0]
    counts_prediction_entropy = counts_prediction_post[counts_prediction_post > 0]

    rmse_error = stats.entropy(counts_data_entropy, counts_prediction_entropy)

    with open(f"{dataset}/model_results/output.txt", "a") as f:
        f.write(
            f"\n***: **{dataset}: the distance of {title} using model {model_name} is {rmse_error}!****\n"
        )


def q_qplot(dataset, data, prediction, title, model_name):
    plt.scatter(data, prediction)
    plt.xlabel("Data")
    plt.ylabel("Prediction")
    plt.savefig(f"{dataset}/model_results/{model_name}{title}components.png")
    plt.show()
    plt.close()
