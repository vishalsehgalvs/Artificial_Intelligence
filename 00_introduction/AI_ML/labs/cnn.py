"""Small 8x8 digit CNN: python labs/cnn.py. Educational data, not production OCR."""

import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def run():
    torch.manual_seed(42)
    digits = load_digits()
    images = torch.tensor(digits.images[:, None] / 16.0, dtype=torch.float32)
    labels = torch.tensor(digits.target, dtype=torch.long)
    train_x, test_x, train_y, test_y = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(8 * 4 * 4, 10),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    assert model(train_x[:2]).shape == (2, 10)
    for epoch in range(12):
        model.train()
        order = torch.randperm(len(train_x))
        for batch in order.split(128):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(train_x[batch]), train_y[batch])
            loss.backward()
            optimizer.step()
        print("Epoch", epoch + 1, "training batch loss", round(loss.item(), 3))
    model.eval()
    with torch.no_grad():
        guesses = model(test_x).argmax(dim=1)
        accuracy = (guesses == test_y).float().mean().item()
    print("Held-out digit accuracy:", round(accuracy, 3))
    print("First five wrong digits (actual, guess):", list(zip(test_y[guesses != test_y][:5].tolist(), guesses[guesses != test_y][:5].tolist())))
    assert len(guesses) == len(test_y) and accuracy > 0.7


if __name__ == "__main__":
    run()