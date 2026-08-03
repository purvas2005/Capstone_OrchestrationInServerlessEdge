import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import (
    LOSS_FUNCTION,
    PREDICTION_HORIZON,
    REQUEST_MAE_LOSS_SCALE,
    REQUEST_MAE_LOSS_WEIGHT,
    OCCURRENCE_LOSS_WEIGHT,
    POSITIVE_COUNT_LOSS_WEIGHT,
)

# ==========================================================
# Gaussian Negative Log Likelihood
# ==========================================================

class GaussianNLLLoss(nn.Module):
    """
    Hurdle loss for probabilistic forecasting.

    Model predicts

        mu

        sigma

    The occurrence head is trained on every minute.  The Gaussian count head
    is trained only where a request actually occurred, so zeros cannot pull
    positive-count estimates toward zero.
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

        has_demand = target > 0.0
        occurrence_loss = F.binary_cross_entropy_with_logits(
            prediction["occurrence_logit"], has_demand.float()
        )

        if has_demand.any():
            normalized_error = (target[has_demand] - mu[has_demand]) / sigma[has_demand]
            count_loss = (0.5 * normalized_error.square() + torch.log(sigma[has_demand])).mean()

            # Keep direct count-space pressure on the positive-count head.
            request_prediction = torch.expm1(mu[has_demand]).clamp_min(0.0)
            request_target = torch.expm1(target[has_demand]).clamp_min(0.0)
            request_mae = torch.abs(request_prediction - request_target).mean()
        else:
            count_loss = occurrence_loss.new_zeros(())
            request_mae = occurrence_loss.new_zeros(())

        return (
            OCCURRENCE_LOSS_WEIGHT * occurrence_loss
            + POSITIVE_COUNT_LOSS_WEIGHT * count_loss
            + REQUEST_MAE_LOSS_WEIGHT * request_mae / REQUEST_MAE_LOSS_SCALE
        )


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
