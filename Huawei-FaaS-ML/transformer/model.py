import torch
import torch.nn as nn


class TransformerPredictor(nn.Module):

    def __init__(
        self,
        input_size,
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        prediction_horizon=10,
    ):

        super().__init__()

        self.embedding = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, prediction_horizon),
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.encoder(x)

        x = x[:, -1, :]

        return self.head(x)
