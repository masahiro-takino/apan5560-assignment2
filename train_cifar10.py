import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    train_loader = get_data_loader(data_dir="data", batch_size=64, train=True)
    test_loader = get_data_loader(data_dir="data", batch_size=64, train=False)

    model = get_model("CNN").to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=1,
        checkpoint_path="checkpoints/cnn_cifar10.pt",
    )

    test_loss, test_accuracy = evaluate_model(
        model=trained_model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
