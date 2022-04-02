
from typing import Callable

import tqdm.notebook
import torch
from torch.utils.data import DataLoader

from src.utils import run_time


@run_time
def training(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    validation_datalaoder: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss: Callable,
    epochs: int
) -> None:

    device = "cpu"
    if torch.cuda.is_available:
        device = "cuda"

    model = model.to(device)

    for epoch in range(epochs):

        print(f"EPOCH {epoch}".center(20, "-"))

        model.train()
        train_tqdm = tqdm.notebook.tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc="Current train loss", leave=True, unit="Batch", unit_divisor=1)

        train_loss = 0

        for batch, (X, Y) in train_tqdm:

            X = X.to(device)
            Y = Y.to(device)

            output = model(X)
            output = torch.reshape(output, (-1,)).float()

            l = loss(output.float(), Y.float())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_loss += l

            del X, Y, output, l

            train_tqdm.set_description(f"Current train loss {train_loss/(batch+1):.3f}")

        model.eval()
        test_tqdm = tqdm.notebook.tqdm(
            enumerate(validation_datalaoder),
            total=len(validation_datalaoder),
            desc="Current test loss", leave=True, unit="Batch", unit_divisor=1)
        test_loss = 0
        correct = 0

        for batch, (X, Y) in test_tqdm:

            X = X.to(device)
            Y = Y.to(device)

            with torch.no_grad():
                output = model(X)
                output = torch.reshape(output, (-1,)).float()
                test_loss += loss(output, Y)
                correct += ((output > 0.5) == Y).type(torch.float).sum().item()

            del X, Y, output

            test_tqdm.set_description(f"Current test loss {test_loss/(batch+1):.3f}")

        print(f"Correct labeling: {correct:.0f} of {batch+1} images")
