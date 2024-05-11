from torch_geometric.loader import DataLoader
import functools
import torch
import os
from matplotlib import pyplot as plt
import json
from src.utils.utils import return_the_model
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def test(model, rank, test_loader):
    model.eval()
    with torch.no_grad():
        for sampled_data in test_loader:
            sampled_data = sampled_data.to(rank)
            pred = model(sampled_data.x, sampled_data.edge_index, sampled_data.region)


def train(model, rank, train_loader, optimizer, loss_fn, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)

    if sampler:
        sampler.set_epoch(epoch)
    for sampled_data in train_loader:
        sampled_data = sampled_data.to(rank)
        optimizer.zero_grad()
        pred = model(sampled_data.x, sampled_data.edge_index, sampled_data.region)
        loss = loss_fn(pred, sampled_data.edge_attr.ravel().to(pred.device))
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(sampled_data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, ddp_loss[0]), flush=True)


def fsdp_main(
    rank,
    world_size,
    model,
    loss_fn,
    dataloader_train,
    dataloader_test,
    epoches,
    dataset,
    model_name,
    loss_name,
):
    setup(rank, world_size)

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )

    torch.cuda.set_device(rank)

    model = model.to(rank)
    model = FSDP(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    for epoch in range(1, epoches + 1):
        train(model, rank, dataloader_train, optimizer, loss_fn, epoch, sampler=None)
        test(model, rank, dataloader_test)

    dist.barrier()
    states = model.state_dict()
    if rank == 0:
        torch.save(
            states,
            f"{dataset}/data/trained_model/GNNParameteres_{model_name}_{loss_name}.pt",
        )
    cleanup()


if __name__ == "__main__":
    path = os.path.abspath(__file__)
    os.chdir("/".join(path.split(os.sep)[:-3]))

    save_training_process = True
    dataset = "SyntheticNetwork"

    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    # idx = 8
    with open(f"config/configSynthetic/parameters_{idx}.json") as config_file:
        config_params = json.load(config_file)
    loss_name = config_params["loss_name"]
    dataset_type = config_params["dataset_type"]
    model_name = config_params["model_name"]
    epoches = config_params["epoch"]
    # epoches = 5

    model, loss_fn = return_the_model(
        model_name, loss_name, False, device=None, input_channels=3
    )
    train_set_pytorch = torch.load(f"{dataset}/data/trainig_set.pt")
    test_set_pytorch = torch.load(f"{dataset}/data/testing_set.pt")

    dataloader_train = DataLoader(train_set_pytorch, batch_size=8)
    dataloader_test = DataLoader(test_set_pytorch, batch_size=8)

    WORLD_SIZE = torch.cuda.device_count()
    print(f"total GPU number is {WORLD_SIZE}##################")

    mp.spawn(
        fsdp_main,
        args=(
            WORLD_SIZE,
            model,
            loss_fn,
            dataloader_train,
            dataloader_test,
            epoches,
            dataset,
            model_name,
            loss_name,
        ),
        nprocs=WORLD_SIZE,
        join=True,
    )
