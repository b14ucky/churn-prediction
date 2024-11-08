import torch
import torch.nn as nn
import torch.optim as optim
from torchtyping import TensorType
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(n_inputs, 128)
        self.l2 = nn.Linear(128, n_outputs)
        self.relu = nn.ReLU()

        self.loss = nn.CrossEntropyLoss()
        self.optim = None

    def forward(self, x: TensorType[float]) -> TensorType[torch.float32]:
        output = self.relu(self.l1(x))
        output = self.l2(output)
        return output

    def fit(self, dataloader: DataLoader, n_epochs: int, lr: float = 0.001) -> None:
        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.train()

        for epoch in range(n_epochs):

            print(f"Epoch: {epoch + 1}/{n_epochs}:")

            for batch_idx, (X, y) in enumerate(dataloader, 1):

                prediction: TensorType[torch.float32] = self(X)
                loss = self.loss(prediction, y.reshape(-1))

                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

                if batch_idx % 10 == 0:
                    current_loss: float = loss.item()
                    print(f"loss: {current_loss}")

    def score(self, dataloader: DataLoader) -> TensorType[torch.float]:
        size: int = len(dataloader.dataset)
        correct: int = 0
        self.eval()

        with torch.no_grad():
            for X, y in dataloader:
                prediction: TensorType[torch.float32] = self(X)

                correct += (prediction.argmax(1) == y.reshape(-1)).type(torch.float).sum().item()

        return correct / size
