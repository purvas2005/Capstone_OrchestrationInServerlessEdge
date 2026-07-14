import math

import torch
import torch.nn as nn

from .config import *


# ==========================================================
# Positional Encoding
# ==========================================================

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """

    def __init__(
        self,
        d_model,
        max_length=5000
    ):

        super().__init__()

        pe = torch.zeros(
            max_length,
            d_model
        )

        position = torch.arange(
            0,
            max_length,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(

            torch.arange(
                0,
                d_model,
                2
            ).float()

            *

            (

                -math.log(10000.0)

                /

                d_model

            )

        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        x = x + self.pe[:, :x.size(1)]

        return x


# ==========================================================
# Transformer Encoder
# ==========================================================

class ForecastEncoder(nn.Module):

    """
    Encoder used for learning historical workload
    representations.

    Input

        (B,L,D_MODEL)

    Output

        (B,L,D_MODEL)
    """

    def __init__(self):

        super().__init__()

        self.position = PositionalEncoding(
            D_MODEL
        )

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=D_MODEL,

            nhead=NHEAD,

            dim_feedforward=DIM_FEEDFORWARD,

            dropout=DROPOUT,

            activation="gelu",

            batch_first=True,

            norm_first=True

        )

        self.encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=NUM_ENCODER_LAYERS

        )

        self.norm = nn.LayerNorm(
            D_MODEL
        )

    def forward(

        self,

        encoder_input,

        padding_mask=None

    ):

        x = self.position(

            encoder_input

        )

        x = self.encoder(

            src=x,

            src_key_padding_mask=padding_mask

        )

        x = self.norm(

            x

        )

        return x


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    batch = 8

    x = torch.randn(

        batch,

        SEQUENCE_LENGTH,

        D_MODEL

    )

    encoder = ForecastEncoder()

    y = encoder(x)

    print()

    print("=" * 60)

    print("Encoder Test")

    print("=" * 60)

    print()

    print("Input")

    print(x.shape)

    print()

    print("Output")

    print(y.shape)
