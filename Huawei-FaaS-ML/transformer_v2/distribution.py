"""Conversions for the zero-inflated lognormal workload distribution."""

import torch


def hurdle_request_quantile(mu, sigma, probability, quantile):
    """Return a request-count quantile for P(Y=0)=1-probability.

    Conditional on demand, ``log1p(Y)`` is Gaussian with parameters ``mu`` and
    ``sigma``.  This makes the reported quantiles consistent with the explicit
    zero-demand probability, rather than treating a zero-heavy workload as one
    continuous lognormal distribution.
    """
    probability = probability.clamp(1e-6, 1.0 - 1e-6)
    conditional_quantile = (quantile - (1.0 - probability)) / probability
    positive = conditional_quantile > 0.0
    conditional_quantile = conditional_quantile.clamp(1e-6, 1.0 - 1e-6)
    z_score = torch.special.ndtri(conditional_quantile)
    positive_value = torch.expm1(mu + sigma * z_score).clamp_min(0.0)
    return torch.where(positive, positive_value, torch.zeros_like(positive_value))


def hurdle_request_mean(mu, sigma, probability):
    """Unconditional request-count mean of the hurdle distribution."""
    conditional_mean = torch.expm1(mu + 0.5 * sigma.square()).clamp_min(0.0)
    return probability * conditional_mean

