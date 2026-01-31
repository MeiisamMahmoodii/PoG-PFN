"""Training package initialization"""

from .losses import (
    EffectLoss,
    GraphPosteriorLoss,
    ClaimValidationLoss,
    IdentificationConsistencyLoss,
    PoGPFNLoss
)

__all__ = [
    'EffectLoss',
    'GraphPosteriorLoss',
    'ClaimValidationLoss',
    'IdentificationConsistencyLoss',
    'PoGPFNLoss'
]
