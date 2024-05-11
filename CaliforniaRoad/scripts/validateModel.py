# %%
import os
import numpy as np
import sys

path = os.path.abspath(__file__)
root_path = "/".join(path.split(os.sep)[:-3])
os.chdir(root_path)
sys.path.append(root_path)

import pickle
import random
import torch

from src.utils.test_setevaluation import (
    count_parameters,
    sample_validation,
    hist_plot,
    bar_plot,
)
from src.utils.utils import return_the_model
import json
from datetime import datetime
from src.utils.evaluation import PerformanceEvaluation


if __name__ == "__main__":
    # define some global parameters that are important for the model
    dataset = "CaliforniaRoad"
    preparation = "region"

    # idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    idx = 2

    with open(f"{dataset}/config/parameters_{idx}.json") as config_file:
        config_params = json.load(config_file)
    model_name = config_params["model_name"]
    loss_name = config_params["loss_name"]
    dataset_type = config_params["dataset_type"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(
        f"{dataset}/data/sampled_nodes{dataset_type}{preparation}.dat", "rb"
    ) as f:
        graphlist = pickle.load(f)

    cut = int(0.7 * len(graphlist))
    random.Random(8).shuffle(graphlist)
    training_set = graphlist[:cut]
    test_set = graphlist[cut:]

    model, loss_fn = return_the_model(
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
    print(f"model parameters {count_parameters(model)}")

    sample_validation(
        dataset, model_name, model, threshold=0.5, test_set=training_set, device=device
    )

    case_evaluation = PerformanceEvaluation(
        dataset, model, model_name, device, training_set
    )

    (
        data_length,
        predict_length,
        data_node,
        predict_node,
        f1_scores,
        roc_scores,
        data_components,
        prediction_components,
    ) = case_evaluation.performance_summary()

    hist_plot(dataset, data_length, predict_length, "Edge length", model_name)
    bar_plot(dataset, data_node, predict_node, "Node degree", model_name)
    hist_plot(
        dataset, f1_scores, prediction=None, title="f1 score", model_name=model_name
    )
    bar_plot(
        dataset,
        data_components,
        prediction_components,
        "Number of Components",
        model_name,
    )

    final = np.mean(np.array(f1_scores))
    final_roc = np.mean(np.array(roc_scores))
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    with open(f"{dataset}/model_results/output.txt", "a") as f:
        f.write(
            f"\n***{dt_string}: **{dataset}: the accuracy of {model_name}, with evaluation: f1_score is {final}, roc score is {final_roc}!****\n"
        )

        # break

        # predict_whole_test(test_set, 0.5, model)
