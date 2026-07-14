import torch
import torch.nn as nn

from .config import *


# ==========================================================
# Numerical Feature Projection
# ==========================================================

class FeatureProjection(nn.Module):
    """
    Projects engineered numerical features into
    Transformer embedding space.
    """

    def __init__(self):

        super().__init__()

        input_dim = len(PAST_VALUE_FEATURES)

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                D_MODEL
            ),

            nn.LayerNorm(
                D_MODEL
            ),

            nn.GELU(),

            nn.Dropout(
                DROPOUT
            )

        )

    def forward(self, x):

        return self.network(x)


# ==========================================================
# Time Embedding
# ==========================================================

class TimeEmbedding(nn.Module):
    """
    Projects cyclical time features into D_MODEL.
    """

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(

                len(TIME_FEATURES),

                D_MODEL

            ),

            nn.LayerNorm(
                D_MODEL
            ),

            nn.GELU(),

            nn.Dropout(
                DROPOUT
            )

        )

    def forward(self, x):

        return self.network(x)


# ==========================================================
# Static Metadata Embedding
# ==========================================================

class StaticEmbedding(nn.Module):

    """
    Learns embeddings for

    Function

    Region

    Cluster

    Category

    Stability
    """

    def __init__(

        self,

        num_functions,

        num_regions,

        num_clusters,

        num_categories,

        num_stability

    ):

        super().__init__()

        self.function_embedding = nn.Embedding(

            num_functions,

            FUNCTION_EMBED_DIM

        )

        self.region_embedding = nn.Embedding(

            num_regions,

            REGION_EMBED_DIM

        )

        self.cluster_embedding = nn.Embedding(

            num_clusters,

            CLUSTER_EMBED_DIM

        )

        self.category_embedding = nn.Embedding(

            num_categories,

            CATEGORY_EMBED_DIM

        )

        self.stability_embedding = nn.Embedding(

            num_stability,

            STABILITY_EMBED_DIM

        )

        self.projection = nn.Sequential(

            nn.Linear(

                STATIC_EMBED_DIM,

                D_MODEL

            ),

            nn.LayerNorm(

                D_MODEL

            ),

            nn.GELU(),

            nn.Dropout(

                DROPOUT

            )

        )

    def forward(

        self,

        function,

        region,

        cluster,

        category,

        stability

    ):

        function_embedding = self.function_embedding(

            function

        )

        region_embedding = self.region_embedding(

            region

        )

        cluster_embedding = self.cluster_embedding(

            cluster

        )

        category_embedding = self.category_embedding(

            category

        )

        stability_embedding = self.stability_embedding(

            stability

        )

        combined = torch.cat(

            [

                function_embedding,

                region_embedding,

                cluster_embedding,

                category_embedding,

                stability_embedding

            ],

            dim=-1

        )

        return self.projection(

            combined

        )
# ==========================================================
# Complete Input Embedding
# ==========================================================


class InputEmbedding(nn.Module):
    """
    Creates the encoder input.

    Inputs

        past_values
            (B, L, num_features)

        past_time_features
            (B, L, 4)

        function
        region
        cluster
        category
        stability

    Outputs

        encoder_input
            (B, L, D_MODEL)

        static_embedding
            (B, D_MODEL)
    """

    def __init__(

        self,

        num_functions,

        num_regions,

        num_clusters,

        num_categories,

        num_stability

    ):

        super().__init__()

        self.feature_projection = FeatureProjection()

        self.time_embedding = TimeEmbedding()

        self.static_embedding = StaticEmbedding(

            num_functions=num_functions,

            num_regions=num_regions,

            num_clusters=num_clusters,

            num_categories=num_categories,

            num_stability=num_stability

        )

    # ------------------------------------------------------

    def forward(

        self,

        past_values,

        past_time_features,

        function,

        region,

        cluster,

        category,

        stability

    ):

        # ----------------------------------------------
        # Numerical features
        # ----------------------------------------------

        value_embedding = self.feature_projection(

            past_values

        )

        # ----------------------------------------------
        # Time embedding
        # ----------------------------------------------

        time_embedding = self.time_embedding(

            past_time_features

        )

        # ----------------------------------------------
        # Static metadata
        # ----------------------------------------------

        static_embedding = self.static_embedding(

            function,

            region,

            cluster,

            category,

            stability

        )

        # ----------------------------------------------
        # Expand static embedding across sequence
        # ----------------------------------------------

        expanded_static = static_embedding.unsqueeze(1)

        expanded_static = expanded_static.expand(

            -1,

            value_embedding.size(1),

            -1

        )

        # ----------------------------------------------
        # Final encoder input
        # ----------------------------------------------

        encoder_input = (

            value_embedding +

            time_embedding +

            expanded_static

        )

        return (

            encoder_input,

            static_embedding

        )


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    batch = 8

    sequence = SEQUENCE_LENGTH

    values = torch.randn(

        batch,

        sequence,

        len(PAST_VALUE_FEATURES)

    )

    time = torch.randn(

        batch,

        sequence,

        len(TIME_FEATURES)

    )

    function = torch.randint(

        0,

        4267,

        (batch,)

    )

    region = torch.randint(

        0,

        5,

        (batch,)

    )

    cluster = torch.randint(

        0,

        4,

        (batch,)

    )

    category = torch.randint(

        0,

        3,

        (batch,)

    )

    stability = torch.randint(

        0,

        2,

        (batch,)

    )

    embedding = InputEmbedding(

        num_functions=4267,

        num_regions=5,

        num_clusters=4,

        num_categories=3,

        num_stability=2

    )

    encoder_input, static_embedding = embedding(

        values,

        time,

        function,

        region,

        cluster,

        category,

        stability

    )

    print()

    print("=" * 60)

    print("Embedding Test")

    print("=" * 60)

    print()

    print("Encoder Input")

    print(

        encoder_input.shape

    )

    print()

    print("Static Embedding")

    print(

        static_embedding.shape

    )
