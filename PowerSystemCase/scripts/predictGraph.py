# %%
from torch_geometric.loader import DataLoader
import torch
import tqdm
import os
import sys

path = os.path.abspath(__file__)
root_path = "/".join(path.split(os.sep)[:-3])
os.chdir(root_path)
sys.path.append(root_path)
# print(os.getcwd())
from matplotlib import pyplot as plt
import json

from src.utils.utils import return_the_model


@torch.no_grad()
def test(data, model, loss, model_name):
    model.eval()
    pred = model(data.x, data.edge_index)
    loss = loss_fn(pred, data.edge_attr.ravel().to(pred.device))
    num_example = pred.numel()
    return loss, num_example


if __name__ == "__main__":
    print(f"current directory is {os.getcwd()}")
    save_training_process = True
    dataset = "PowerSystemCase"
    preparation = "region"

    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    # idx = 11
    with open(f"{dataset}/config/parameters_{idx}.json") as config_file:
        config_params = json.load(config_file)
    model_name = config_params["model_name"]
    loss_name = config_params["loss_name"]
    dataset_type = config_params["dataset_type"]
    epoches = config_params["epoch"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set_pytorch = torch.load(
        f"{dataset}/data/trainig_set{dataset_type}{preparation}.pt"
    )
    test_set_pytorch = torch.load(
        f"{dataset}/data/testing_set{dataset_type}{preparation}.pt"
    )

    dataloader_train = DataLoader(train_set_pytorch, batch_size=32)
    dataloader_test = DataLoader(test_set_pytorch, batch_size=32)
    model, loss_fn = return_the_model(
        model_name, loss_name, False, device, input_channels=4
    )
    total_params = sum(param.numel() for param in model.parameters())

    print(f"total model parameters is {total_params}")
    print(f"Training samples is {len(dataloader_train.dataset)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    training_loss_total = []
    testing_loss_total = []

    for epoch in range(0, int(epoches)):
        total_loss = total_examples = 0
        test_acc = test_examples = 0
        for sampled_data in tqdm.tqdm(dataloader_train):
            optimizer.zero_grad()
            sampled_data.to(device)
            pred = model(sampled_data.x, sampled_data.edge_index)
            loss = loss_fn(pred, sampled_data.edge_attr.ravel().to(pred.device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        training_loss = total_loss / total_examples
        training_loss_total.append(training_loss)

        print(f"Epoch: {epoch:03d}, Loss: {training_loss:.4f}")

        for sampled_test in tqdm.tqdm(dataloader_test):
            sampled_test.to("cuda:0")
            test_loss, sample_num = test(sampled_test, model, loss_fn, model_name)
            test_acc += float(test_loss) * sample_num
            test_examples += sample_num
        testing_loss = test_acc / test_examples
        testing_loss_total.append(testing_loss)
        print(f"Epoch: {epoch:03d}, Test loss: {testing_loss :.4f}")

        if epoch % 10 == 0 and save_training_process:
            plt.plot(training_loss_total, label="train_loss")
            plt.plot(testing_loss_total, label="val_loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(
                f"{dataset}/model_results/model_training_process_{model_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    torch.save(
        model.state_dict(),
        f"{dataset}/data/trained_model/GNNParameteres_{model_name}.pt",
    )
