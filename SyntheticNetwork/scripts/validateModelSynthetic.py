# %%
import os
import numpy as np

path = os.path.abspath(__file__)
os.chdir("/".join(path.split(os.sep)[:-3]))
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
    dataset = "SyntheticNetwork"
    evaluation = "f1_score"

    # idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    for idx in range(3):
        with open(f"{dataset}/config/parameters_{idx}.json") as config_file:
            config_params = json.load(config_file)
        model_name = config_params["model_name"]
        loss_name = config_params["loss_name"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(f"{dataset}/data/sampled_nodes.dat", "rb") as f:
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
            input_channels=3,
        )
        model.load_state_dict(
            torch.load(
                f"{dataset}/data/trained_model/GNNParameteres_{model_name}_{loss_name}.pt",
                map_location="cpu",
            )
        )

        model.to(device)
        print(f"model parameters {count_parameters(model)}")

        sample_validation(
            dataset,
            model_name,
            model,
            threshold=0.5,
            test_set=test_set,
            device=device,
            seed=128,
        )
        # case_evaluation = PerformanceEvaluation(
        #     dataset, model, model_name, device, test_set
        # )

        # (
        #     data_length,
        #     predict_length,
        #     data_node,
        #     predict_node,
        #     f1_scores,
        #     roc_scores,
        #     data_components,
        #     prediction_components,
        # ) = case_evaluation.performance_summary()

        # hist_plot(dataset, data_length, predict_length, "Edge length", model_name)
        # bar_plot(dataset, data_node, predict_node, "Node degree", model_name)
        # hist_plot(
        #     dataset, f1_scores, prediction=None, title="f1 score", model_name=model_name
        # )
        # bar_plot(
        #     dataset,
        #     data_components,
        #     prediction_components,
        #     "Number of Components",
        #     model_name,
        # )

        # final = np.mean(np.array(f1_scores))
        # final_roc = np.mean(np.array(roc_scores))
        # now = datetime.now()
        # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        # with open(f"{dataset}/model_results/output.txt", "a") as f:
        #     f.write(
        #         f"\n***{dt_string}: **{dataset}: the accuracy of {model_name}, with evaluation: f1_score is {final}, roc score is {final_roc}!****\n"
        #     )
