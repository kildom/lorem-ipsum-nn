import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from datetime import datetime


DEFAULT_LEARNING_RATE = 3e-3
BATCH_SIZE = 64
DEFAULT_EPOCHS = 20


def train(model: nn.Module, dataset: Dataset, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LEARNING_RATE, epoch_callback=None, epoch_offset=0, skip_first_train=False):
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # TensorBoard setup
    log_dir = f"runs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    progress_bar = tqdm(total=len(train_loader) * ((epochs - 1) if skip_first_train else epochs) + len(val_loader) * epochs)
    progress_value = 0
    progress_old = 0

    for epoch in range(epochs):
        if not skip_first_train:
            model.train()
            total_loss = 0
            correct = 0
            for x, y in train_loader:
                progress_value += 1
                if progress_value % 100 == 0:
                    progress_bar.update(progress_value - progress_old)
                    progress_old = progress_value

                logits = model(x)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                correct += (logits.argmax(dim=1) == y).sum().item()

            acc = correct / train_size
            progress_bar.clear()
            print(f"Epoch {epoch + 1 + epoch_offset:3d} - Train Loss: {total_loss/train_size:.4f}, Acc: {acc:.4f}")
            progress_bar.update(0)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                progress_value += 1
                if progress_value % 100 == 0:
                    progress_bar.update(progress_value - progress_old)
                    progress_old = progress_value
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (logits.argmax(dim=1) == y).sum().item()

        val_acc = val_correct / val_size
        progress_bar.clear()
        print(f"          - Val Loss: {val_loss/val_size:.4f}, Acc: {val_acc:.4f}")
        progress_bar.update(0)

        if not skip_first_train:
            writer.add_scalars("Loss", {"train": total_loss/train_size, "val": val_loss/val_size}, epoch + epoch_offset)
            writer.add_scalars("Accuracy", {"train": acc, "val": val_acc}, epoch + epoch_offset)
        else:
            writer.add_scalars("Loss", {"val": val_loss/val_size}, epoch + epoch_offset)
            writer.add_scalars("Accuracy", {"val": val_acc}, epoch + epoch_offset)

        writer.flush()

        if epoch_callback is not None:
            progress_bar.clear()
            if not skip_first_train:
                epoch_callback(model, epoch, epochs, total_loss/train_size, val_loss/val_size, acc, val_acc)
            else:
                epoch_callback(model, epoch, epochs, None, val_loss/val_size, None, val_acc)
            progress_bar.update(0)

        skip_first_train = False

    writer.close()

