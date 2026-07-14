import torch
import torch.nn as nn

from .config import *
from .encoder import PositionalEncoding
from .embeddings import TimeEmbedding


# ==========================================================
# Forecast Decoder
# ==========================================================

class ForecastDecoder(nn.Module):
    """
    Transformer Decoder.

    Inputs

        future_time_features
            (B,H,4)

        encoder_memory
            (B,L,D_MODEL)

        static_embedding
            (B,D_MODEL)

    Output

        (B,H,D_MODEL)
    """

    def __init__(self):

        super().__init__()

        self.time_embedding = TimeEmbedding()

        self.position = PositionalEncoding(
            D_MODEL
        )

        decoder_layer = nn.TransformerDecoderLayer(

            d_model=D_MODEL,

            nhead=NHEAD,

            dim_feedforward=DIM_FEEDFORWARD,

            dropout=DROPOUT,

            activation="gelu",

            batch_first=True,

            norm_first=True

        )

        self.decoder = nn.TransformerDecoder(

            decoder_layer,

            NUM_DECODER_LAYERS

        )

        self.norm = nn.LayerNorm(
            D_MODEL
        )

    # ------------------------------------------------------

    def forward(

        self,

        future_time_features,

        encoder_memory,

        static_embedding

    ):

        # -----------------------------------------
        # Embed future timestamps
        # -----------------------------------------

        x = self.time_embedding(

            future_time_features

        )

        # -----------------------------------------
        # Broadcast static embedding
        # -----------------------------------------

        static_embedding = static_embedding.unsqueeze(1)

        static_embedding = static_embedding.expand(

            -1,

            x.size(1),

            -1

        )

        # -----------------------------------------
        # Add metadata context
        # -----------------------------------------

        x = x + static_embedding

        # -----------------------------------------
        # Positional Encoding
        # -----------------------------------------

        x = self.position(x)

        # -----------------------------------------
        # Decoder
        # -----------------------------------------

        x = self.decoder(

            tgt=x,

            memory=encoder_memory

        )

        x = self.norm(x)

        return x


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    batch = 4

    memory = torch.randn(

        batch,

        SEQUENCE_LENGTH,

        D_MODEL

    )

    future_time = torch.randn(

        batch,

        PREDICTION_HORIZON,

        len(TIME_FEATURES)

    )

    static = torch.randn(

        batch,

        D_MODEL

    )

    decoder = ForecastDecoder()

    output = decoder(

        future_time,

        memory,

        static

    )

    print()

    print("=" * 60)

    print("Decoder Test")

    print("=" * 60)

    print()

    print("Memory")

    print(memory.shape)

    print()

    print("Future Time")

    print(future_time.shape)

    print()

    print("Output")

    print(output.shape)
