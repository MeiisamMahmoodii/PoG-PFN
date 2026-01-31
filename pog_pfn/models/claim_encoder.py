"""
Module B: Claim Encoder

Processes partial causal claims/constraints into embeddings.
Claims are represented as structured tokens with relation types, variables involved,
and user confidence scores.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from enum import Enum


class RelationType(Enum):
    """Types of causal relations that can be claimed."""
    CAUSES = 0  # X → Y (directed edge)
    FORBIDS = 1  # X ↛ Y (no directed path)
    CONFOUNDER = 2  # Z → X and Z → Y
    MEDIATOR = 3  # X → ... → Z → ... → Y
    INSTRUMENT = 4  # Z → X, Z ↛ Y (except through X)
    COLLIDER = 5  # X → Z ← Y
    ANCESTOR = 6  # X ⤳ Y (directed path exists)
    NON_ANCESTOR = 7  # X ⤳ Y (no directed path)
    INDEPENDENT = 8  # X ⊥ Y | ∅
    COND_INDEPENDENT = 9  # X ⊥ Y | Z


class Claim:
    """
    Structured representation of a causal claim.
    
    Attributes:
        var_a: First variable index
        var_b: Second variable index (if binary relation)
        var_c: Third variable index (optional, e.g., for conditioning)
        relation_type: Type of relation (RelationType)
        confidence: User confidence in [0, 1]
        is_true: Ground truth (for training only)
    """
    
    def __init__(
        self,
        var_a: int,
        var_b: Optional[int] = None,
        var_c: Optional[int] = None,
        relation_type: RelationType = RelationType.CAUSES,
        confidence: float = 1.0,
        is_true: bool = True
    ):
        self.var_a = var_a
        self.var_b = var_b
        self.var_c = var_c
        self.relation_type = relation_type
        self.confidence = confidence
        self.is_true = is_true
    
    def to_tensor(self, n_relation_types: int = 10, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Convert claim to tensor representation."""
        return {
            'var_a': torch.tensor([self.var_a], dtype=torch.long, device=device),
            'var_b': torch.tensor([self.var_b if self.var_b is not None else -1], dtype=torch.long, device=device),
            'var_c': torch.tensor([self.var_c if self.var_c is not None else -1], dtype=torch.long, device=device),
            'relation_type': torch.tensor([self.relation_type.value], dtype=torch.long, device=device),
            'confidence': torch.tensor([self.confidence], dtype=torch.float, device=device),
        }


class ClaimEncoder(nn.Module):
    """
    Encodes causal claims into embeddings and performs cross-attention
    with variable embeddings from the dataset encoder.
    """
    
    def __init__(
        self,
        n_vars: int,
        d_model: int = 256,
        claim_d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_claims: int = 50,
        n_relation_types: int = 10
    ):
        super().__init__()
        
        self.n_vars = n_vars
        self.d_model = d_model
        self.claim_d_model = claim_d_model
        self.max_claims = max_claims
        self.n_relation_types = n_relation_types
        
        # Embeddings for claim components
        self.variable_embedding = nn.Embedding(n_vars + 1, claim_d_model)  # +1 for padding (-1)
        self.relation_type_embedding = nn.Embedding(n_relation_types, claim_d_model)
        self.confidence_projection = nn.Linear(1, claim_d_model)
        
        # Claim token encoder: combines all claim components
        self.claim_token_encoder = nn.Linear(claim_d_model * 4 + claim_d_model, claim_d_model)
        
        # Transformer over claims
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=claim_d_model,
            nhead=n_heads,
            dim_feedforward=claim_d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.claim_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Cross-attention: claims attend to variable embeddings
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=claim_d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Project to match dataset encoder dimension
        self.output_projection = nn.Linear(claim_d_model, d_model)
        
        # Project variable embeddings to claim dimension for cross-attention
        self.var_to_claim_proj = nn.Linear(d_model, claim_d_model)
        
        # Global claim context projection
        self.claim_context_proj = nn.Linear(claim_d_model, d_model)
        
    def encode_claims(
        self,
        claims: List[List[Claim]],
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode list of claims into tensors.
        
        Args:
            claims: List of claim lists, one per batch item [[Claim, ...], [Claim, ...], ...]
            device: Device to place tensors on
            
        Returns:
            claim_tensors: [batch, max_claims, claim_d_model]
            claim_mask: [batch, max_claims] (True = padding)
        """
        batch_size = len(claims)
        max_claims_in_batch = max(len(c) for c in claims)
        
        # Prepare tensors
        var_a = torch.zeros(batch_size, max_claims_in_batch, dtype=torch.long, device=device)
        var_b = torch.zeros(batch_size, max_claims_in_batch, dtype=torch.long, device=device)
        var_c = torch.zeros(batch_size, max_claims_in_batch, dtype=torch.long, device=device)
        relation_types = torch.zeros(batch_size, max_claims_in_batch, dtype=torch.long, device=device)
        confidences = torch.zeros(batch_size, max_claims_in_batch, dtype=torch.float, device=device)
        mask = torch.ones(batch_size, max_claims_in_batch, dtype=torch.bool, device=device)
        
        for i, claim_list in enumerate(claims):
            for j, claim in enumerate(claim_list):
                var_a[i, j] = claim.var_a
                var_b[i, j] = claim.var_b if claim.var_b is not None else self.n_vars  # Use n_vars as padding
                var_c[i, j] = claim.var_c if claim.var_c is not None else self.n_vars
                relation_types[i, j] = claim.relation_type.value
                confidences[i, j] = claim.confidence
                mask[i, j] = False  # Valid claim
        
        # Embed components
        var_a_emb = self.variable_embedding(var_a)  # [batch, max_claims, claim_d_model]
        var_b_emb = self.variable_embedding(var_b)
        var_c_emb = self.variable_embedding(var_c)
        relation_emb = self.relation_type_embedding(relation_types)
        confidence_emb = self.confidence_projection(confidences.unsqueeze(-1))
        
        # Concatenate and project
        claim_tokens = torch.cat([
            var_a_emb,
            var_b_emb,
            var_c_emb,
            relation_emb,
            confidence_emb
        ], dim=-1)  # [batch, max_claims, 5 * claim_d_model]
        
        claim_tokens = self.claim_token_encoder(claim_tokens)  # [batch, max_claims, claim_d_model]
        
        return claim_tokens, mask
    
    def forward(
        self,
        claims: List[List[Claim]],
        variable_embeddings: torch.Tensor,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode claims and perform cross-attention with variable embeddings.
        
        Args:
            claims: List of claim lists per batch
            variable_embeddings: Variable embeddings from dataset encoder [batch, n_vars, d_model]
            device: Device for computations
            
        Returns:
            claim_embeddings: Processed claim embeddings [batch, max_claims, d_model]
            claim_context: Global claim context [batch, d_model]
            claim_mask: Padding mask [batch, max_claims]
        """
        # Encode claims into tokens
        claim_tokens, claim_mask = self.encode_claims(claims, device)  # [batch, max_claims, claim_d_model]
        
        # Self-attention over claims
        claim_encoded = self.claim_transformer(
            claim_tokens,
            src_key_padding_mask=claim_mask
        )  # [batch, max_claims, claim_d_model]
        
        # Project variable embeddings to claim dimension for cross-attention
        var_emb_proj = self.var_to_claim_proj(variable_embeddings)  # [batch, n_vars, claim_d_model]
        
        # Cross-attention: claims attend to variables
        claim_attended, _ = self.cross_attention(
            query=claim_encoded,
            key=var_emb_proj,
            value=var_emb_proj,
            key_padding_mask=None  # All variables are valid
        )  # [batch, max_claims, claim_d_model]
        
        # Combine self-attention and cross-attention
        claim_combined = claim_encoded + claim_attended
        
        # Project back to d_model
        claim_embeddings = self.output_projection(claim_combined)  # [batch, max_claims, d_model]
        
        # Global claim context (mean pooling over valid claims)
        claim_context_raw = claim_combined.masked_fill(claim_mask.unsqueeze(-1), 0).sum(dim=1)
        n_valid = (~claim_mask).sum(dim=1, keepdim=True).clamp(min=1)
        claim_context = claim_context_raw / n_valid  # [batch, claim_d_model]
        claim_context = self.claim_context_proj(claim_context)  # [batch, d_model]
        
        return claim_embeddings, claim_context, claim_mask


if __name__ == "__main__":
    # Test the module
    batch_size = 4
    n_vars = 10
    d_model = 128
    
    # Create dummy claims
    claims = [
        [
            Claim(0, 1, None, RelationType.CAUSES, 0.9),
            Claim(2, 3, None, RelationType.FORBIDS, 0.7),
            Claim(4, 5, 6, RelationType.CONFOUNDER, 0.8),
        ],
        [
            Claim(1, 2, None, RelationType.MEDIATOR, 0.95),
        ],
        [
            Claim(0, 1, None, RelationType.CAUSES, 1.0),
            Claim(3, 4, None, RelationType.INSTRUMENT, 0.6),
        ],
        []  # Empty claims
    ]
    
    # Create dummy variable embeddings
    var_embeddings = torch.randn(batch_size, n_vars, d_model)
    
    # Create encoder
    encoder = ClaimEncoder(
        n_vars=n_vars,
        d_model=d_model,
        claim_d_model=64,
        n_heads=4,
        n_layers=2
    )
    
    # Forward pass
    claim_embs, claim_ctx, claim_mask = encoder(claims, var_embeddings, device='cpu')
    
    print(f"Claim embeddings shape: {claim_embs.shape}")  # [4, 3, 128]
    print(f"Claim context shape: {claim_ctx.shape}")  # [4, 128]
    print(f"Claim mask shape: {claim_mask.shape}")  # [4, 3]
    print("✓ Claim encoder test passed!")
