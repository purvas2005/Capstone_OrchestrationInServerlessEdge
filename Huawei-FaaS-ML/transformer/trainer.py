import torch


def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    running_loss = 0.0

    for batch_idx, batch in enumerate(loader):

        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()

        prediction = model(x)

        loss = criterion(prediction, y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 50 == 0:
            print(
                f"Batch {batch_idx}/{len(loader)} | Loss = {loss.item():.6f}",
                flush=True
            )

    return running_loss / len(loader)
def evaluate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for batch in loader:

            x = batch["x"].to(device)
            y = batch["y"].to(device)

            prediction = model(x)

            loss = criterion(prediction, y)

            running_loss += loss.item()

    return running_loss / len(loader)
