import torch
import torch.nn as nn
from .config import (
    LOSS_FUNCTION,
    PREDICTION_HORIZON,
    REQUEST_MAE_LOSS_SCALE,
    REQUEST_MAE_LOSS_WEIGHT,
)

# ==========================================================
# Gaussian Negative Log Likelihood
# ==========================================================

class GaussianNLLLoss(nn.Module):
    """
    Loss for probabilistic forecasting.

    Model predicts

        mu

        sigma

    Loss = -log p(y | mu, sigma)
    """

    def __init__(self):

        super().__init__()

    def forward(

        self,

        prediction,

        target

    ):

        mu = prediction["mu"]

        sigma = prediction["sigma"]

        normalized_error = (target - mu) / sigma

        loss = 0.5 * normalized_error.square() + torch.log(sigma)

        # NLL calibrates the uncertainty interval.  This auxiliary term trains
        # the central forecast on the metric used by the serving workload.
        request_prediction = torch.expm1(mu).clamp_min(0.0)
        request_target = torch.expm1(target).clamp_min(0.0)
        request_mae = torch.abs(request_prediction - request_target).mean()

        return loss.mean() + REQUEST_MAE_LOSS_WEIGHT * request_mae / REQUEST_MAE_LOSS_SCALE


# ==========================================================
# Mean Squared Error
# ==========================================================

class ForecastMSELoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.loss = nn.MSELoss()

    def forward(

        self,

        prediction,

        target

    ):

        return self.loss(

            prediction["mu"],

            target

        )


# ==========================================================
# Quantile Loss
# ==========================================================

class QuantileLoss(nn.Module):

    def __init__(

        self,

        quantile=0.5

    ):

        super().__init__()

        self.q = quantile

    def forward(

        self,

        prediction,

        target

    ):

        prediction = prediction["mu"]

        error = target - prediction

        loss = torch.maximum(

            self.q * error,

            (self.q - 1) * error

        )

        return loss.mean()


# ==========================================================
# Factory
# ==========================================================

def get_loss():

    if LOSS_FUNCTION.lower() == "gaussian":

        return GaussianNLLLoss()

    elif LOSS_FUNCTION.lower() == "mse":

        return ForecastMSELoss()

    elif LOSS_FUNCTION.lower() == "quantile":

        return QuantileLoss()

    else:

        raise ValueError(

            f"Unknown loss function {LOSS_FUNCTION}"

        )


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    prediction = {

        "mu": torch.randn(

            16,

            PREDICTION_HORIZON

        ),

        "sigma": torch.rand(

            16,

            PREDICTION_HORIZON

        ) + 0.1

    }

    target = torch.randn(

        16,

        PREDICTION_HORIZON

    )

    criterion = get_loss()

    loss = criterion(

        prediction,

        target

    )

    print()

    print("=" * 60)

    print("Loss Test")

    print("=" * 60)

    print()

    print(loss)
