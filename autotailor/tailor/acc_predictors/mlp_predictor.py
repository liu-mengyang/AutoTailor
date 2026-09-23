# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import numpy as np
import torch
import torch.nn as nn

__all__ = ["MLPPredictor"]


class MLPPredictor(nn.Module):
    def __init__(
        self,
        arch_encoder,
        hidden_size=400,
        n_layers=3,
        checkpoint_path=None,
        device="cuda:0",
    ):
        super(MLPPredictor, self).__init__()
        self.arch_encoder = arch_encoder
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        self.split_metadata = None

        # build layers
        layers = []
        for i in range(self.n_layers):
            layers.append(
                nn.Sequential(
                    nn.Linear(
                        self.arch_encoder.n_dim if i == 0 else self.hidden_size,
                        self.hidden_size,
                    ),
                    nn.ReLU(inplace=True),
                )
            )
        layers.append(nn.Linear(self.hidden_size, 1, bias=False))
        self.layers = nn.Sequential(*layers)
        self.base_acc = nn.Parameter(
            torch.zeros(1, device=self.device), requires_grad=False
        )

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.split_metadata = checkpoint.get("split_metadata")
            self.load_state_dict(checkpoint.get("state_dict", checkpoint))
            print("Loaded checkpoint from %s" % checkpoint_path)

        self.layers = self.layers.to(self.device)

    def forward(self, x):
        y = self.layers(x).squeeze()
        return y + self.base_acc

    def predict_accuracy(self, sample):
        X = self.arch_encoder.encode(sample)
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        self.eval()
        with torch.no_grad():
            pred = self.forward(X)
            # print(pred)
            return float(pred.item() * 100.0)
