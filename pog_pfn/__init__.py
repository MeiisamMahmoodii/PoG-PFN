"""
PoG-PFN: Posterior-over-Graphs Prior-Fitted Network

A transformer-based architecture for causal effect estimation with claim-conditioned
Bayesian updating over causal structures and differentiable causal identification.
"""

from .models.pog_pfn import PoGPFN
from .data.scm_generator import SCMGenerator
from .data.claim_generator import ClaimGenerator

__version__ = "0.1.0"
__all__ = ["PoGPFN", "SCMGenerator", "ClaimGenerator"]
