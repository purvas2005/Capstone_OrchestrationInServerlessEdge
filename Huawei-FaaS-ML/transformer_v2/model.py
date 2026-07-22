import torch
import torch.nn as nn

from .config import *

from .embeddings import InputEmbedding
from .encoder import ForecastEncoder
from .decoder import ForecastDecoder


# ==========================================================
# Distribution Head
# ==========================================================

class DistributionHead(nn.Module):
    """
    Predicts

        μ (mean)

        σ (standard deviation)

    for every future timestep.
    """

    def __init__(self):

        super().__init__()

        self.mu_head = nn.Sequential(

            nn.Linear(

                D_MODEL,

                D_MODEL // 2

            ),

            nn.GELU(),

            nn.Linear(

                D_MODEL // 2,

                1

            )

        )

        self.sigma_head = nn.Sequential(

            nn.Linear(

                D_MODEL,

                D_MODEL // 2

            ),

            nn.GELU(),

            nn.Linear(

                D_MODEL // 2,

                1

            ),

            nn.Softplus()

        )

    def forward(self, decoder_output):

        mu = self.mu_head(

            decoder_output

        ).squeeze(-1)

        sigma = self.sigma_head(

            decoder_output

        ).squeeze(-1)

        sigma = sigma + 1e-6

        return {

            "mu": mu,

            "sigma": sigma

        }


# ==========================================================
# Huawei Forecast Transformer
# ==========================================================

class HuaweiForecastTransformer(nn.Module):

    def __init__(

        self,

        num_functions,

        num_regions,

        num_clusters,

        num_categories,

        num_stability

    ):

        super().__init__()

        self.embedding = InputEmbedding(

            num_functions,

            num_regions,

            num_clusters,

            num_categories,

            num_stability

        )

        self.encoder = ForecastEncoder()

        self.decoder = ForecastDecoder()

        self.output_head = DistributionHead()

    # ------------------------------------------------------

    def forward(

        self,

        past_values,

        past_time_features,

        future_time_features,

        past_target,

        function,

        region,

        cluster,

        category,

        stability

    ):

        # -----------------------------------------
        # Input Embedding
        # -----------------------------------------

        encoder_input, static_embedding = self.embedding(

            past_values,

            past_time_features,

            function,

            region,

            cluster,

            category,

            stability

        )

        # -----------------------------------------
        # Encode History
        # -----------------------------------------

        memory = self.encoder(

            encoder_input

        )

        # -----------------------------------------
        # Decode Future
        # -----------------------------------------

        decoder_output = self.decoder(

            future_time_features,

            memory,

            static_embedding

        )

        # -----------------------------------------
        # Distribution Parameters
        # -----------------------------------------

        prediction = self.output_head(

            decoder_output

        )

        last_observation = past_target[:, -1].unsqueeze(-1)
        prediction["mu"] = prediction["mu"] + last_observation

        return prediction


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    batch = 4

    model = HuaweiForecastTransformer(

        num_functions=4267,

        num_regions=5,

        num_clusters=4,

        num_categories=3,

        num_stability=2

    )

    output = model(

        torch.randn(

            batch,

            SEQUENCE_LENGTH,

            len(PAST_VALUE_FEATURES)

        ),

        torch.randn(

            batch,

            SEQUENCE_LENGTH,

            len(TIME_FEATURES)

        ),

        torch.randn(

            batch,

            PREDICTION_HORIZON,

            len(TIME_FEATURES)

        ),

        torch.randn(

            batch,

            SEQUENCE_LENGTH

        ),

        torch.randint(

            0,

            4267,

            (batch,)

        ),

        torch.randint(

            0,

            5,

            (batch,)

        ),

        torch.randint(

            0,

            4,

            (batch,)

        ),

        torch.randint(

            0,

            3,

            (batch,)

        ),

        torch.randint(

            0,

            2,

            (batch,)

        )

    )

    print()

    print("=" * 60)

    print("Model Test")

    print("=" * 60)

    print()

    print("μ")

    print(

        output["mu"].shape

    )

    print()

    print("σ")

    print(

        output["sigma"].shape

    )
