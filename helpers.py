import torch
import numpy as np
import pandas as pd
from typing import Tuple
from torchtyping import TensorType


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame) -> None:
        self.features, self.labels = data.drop(["Churn"], axis=1), data[["Churn"]]
        self.features: TensorType[torch.float32] = torch.from_numpy(
            self.features.to_numpy(dtype=np.float32)
        )
        self.labels: TensorType[torch.long] = torch.from_numpy(
            self.labels.to_numpy(dtype=np.longlong)
        )

    def __getitem__(self, index) -> Tuple[TensorType[torch.float32], TensorType[torch.float32]]:
        return self.features[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)
