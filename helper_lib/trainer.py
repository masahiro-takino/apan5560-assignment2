import torch
from tqdm import tqdm

from helper_lib.checkpoints import save_checkpoint

def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=1,
    checkpoint_path="checkpoints/cnn_cifar10.pt",
):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}, accuracy={accuracy:.4f}")

        save_checkpoint(model, optimizer, epoch + 1, avg_loss, accuracy, checkpoint_path)

    return model
