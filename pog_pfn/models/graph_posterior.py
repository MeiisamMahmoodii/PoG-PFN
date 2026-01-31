"""
Module C: Graph Posterior Head

Predicts soft adjacency matrices conditioned on both dataset and claims.
Implements the key Bayesian update: p(G | D, claims) ∝ p(D | G) · p(G | claims)

Outputs:
- W_data: What the data suggests (via CI signatures)
- W_claim: What the claims push toward  
- W_posterior: Combined posterior over graphs
- Acyclicity constraint to enforce DAG structure
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class GraphPosteriorHead(nn.Module):
    """
    Predicts posterior distribution over causal graphs given data and claims.
    
    This module is the core novelty of PoG-PFN: it learns to perform amortized
    Bayesian inference over graph structures.
    """
    
    def __init__(
        self,
        n_vars: int,
        d_model: int = 256,
        dropout: float = 0.1,
        predict_adjacency: bool = True,
        predict_ancestry: bool = True,
        acyclicity_penalty: float = 1.0,
        use_notears_constraint: bool = True
    ):
        super().__init__()
        
        self.n_vars = n_vars
        self.d_model = d_model
        self.predict_adjacency = predict_adjacency
        self.predict_ancestry = predict_ancestry
        self.acyclicity_penalty = acyclicity_penalty
        self.use_notears_constraint = use_notears_constraint
        
        # Data-driven graph inference
        # Takes dataset embedding + variable embeddings → graph structure suggested by data
        self.data_graph_encoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        
        # Claim-driven graph inference
        # Takes claim context + variable embeddings → graph structure suggested by claims
        self.claim_graph_encoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        
        # Pairwise edge predictor for data-driven graph
        # Edge features are [var_i + context_i, var_j + context_j] = 4*d_model
        self.data_edge_predictor = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        # Pairwise edge predictor for claim-driven graph
        # Edge features are [var_i + context_i, var_j + context_j] = 4*d_model
        self.claim_edge_predictor = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        # Gating mechanism: learns to combine data and claim evidence
        # Conditioned on sample size (larger samples → trust data more)
        # and claim confidence (higher confidence → trust claims more)
        self.gate_network = nn.Sequential(
            nn.Linear(3, 64),  # [n_samples_normalized, avg_claim_confidence, data_uncertainty]
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Ancestry predictor (optional, for longer-range dependencies)
        if predict_ancestry:
            self.ancestry_predictor = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1)
            )
    
    def predict_edges(
        self,
        variable_embeddings: torch.Tensor,
        context: torch.Tensor,
        predictor: nn.Module
    ) -> torch.Tensor:
        """
        Predict pairwise edge probabilities.
        
        Args:
            variable_embeddings: [batch, n_vars, d_model]
            context: [batch, d_model] (dataset or claim context)
            predictor: Edge predictor module
            
        Returns:
            edge_logits: [batch, n_vars, n_vars]
        """
        batch_size = variable_embeddings.shape[0]
        
        # Expand context to match each variable
        context_expanded = context.unsqueeze(1).expand(-1, self.n_vars, -1)  # [batch, n_vars, d_model]
        
        # Combine variable embeddings with context
        var_with_context = torch.cat([variable_embeddings, context_expanded], dim=-1)  # [batch, n_vars, 2*d_model]
        var_encoded = var_with_context  # Could add more processing here
        
        # Compute pairwise edge scores
        # For each pair (i, j), concatenate var_i and var_j embeddings
        var_i = var_encoded.unsqueeze(2).expand(-1, -1, self.n_vars, -1)  # [batch, n_vars, n_vars, d_model*2]
        var_j = var_encoded.unsqueeze(1).expand(-1, self.n_vars, -1, -1)  # [batch, n_vars, n_vars, d_model*2]
        
        # Note: var_i represents source, var_j represents target (i → j)
        edge_features = torch.cat([var_i, var_j], dim=-1)  # [batch, n_vars, n_vars, 4*d_model]
        
        # Predict edge logits
        edge_logits = predictor(edge_features).squeeze(-1)  # [batch, n_vars, n_vars]
        
        # Mask diagonal (no self-loops)
        mask = torch.eye(self.n_vars, device=edge_logits.device, dtype=torch.bool)
        edge_logits = edge_logits.masked_fill(mask.unsqueeze(0), -1e9)
        
        return edge_logits
    
    def compute_acyclicity_loss(self, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Compute NOTEARS acyclicity constraint: h(W) = tr(e^(W ∘ W)) - d
        h(W) = 0 iff W represents a DAG.
        
        Args:
            adjacency: Soft adjacency matrix [batch, n_vars, n_vars]
            
        Returns:
            h: Acyclicity penalty
        """
        # Apply sigmoid to get probabilities in [0, 1]
        W = torch.sigmoid(adjacency)
        
        # NOTEARS: h(W) = tr(e^(W ∘ W)) - d
        W_squared = W * W
        
        # Matrix exponential
        M = torch.matrix_exp(W_squared)
        
        # Trace
        batch_size = M.shape[0]
        h = torch.diagonal(M, dim1=1, dim2=2).sum(dim=1) - self.n_vars
        
        return h.mean()
    
    def forward(
        self,
        dataset_embedding: torch.Tensor,
        variable_embeddings: torch.Tensor,
        claim_context: torch.Tensor,
        n_samples: int,
        avg_claim_confidence: float = 0.5,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Predict posterior over graphs conditioned on data and claims.
        
        Args:
            dataset_embedding: Global dataset representation [batch, d_model]
            variable_embeddings: Per-variable representations [batch, n_vars, d_model]
            claim_context: Global claim context [batch, d_model]
            n_samples: Number of samples in the dataset (for gating)
            avg_claim_confidence: Average confidence of claims (for gating)
            return_components: If True, return W_data, W_claim separately
            
        Returns:
            W_posterior: Posterior adjacency matrix [batch, n_vars, n_vars]
            components: Optional dict with W_data, W_claim, gate, acyclicity_loss
        """
        batch_size = variable_embeddings.shape[0]
        device = variable_embeddings.device
        
        # Encode variable embeddings with data context
        var_data_encoded = self.data_graph_encoder(
            torch.cat([variable_embeddings, dataset_embedding.unsqueeze(1).expand(-1, self.n_vars, -1)], dim=-1)
        )
        
        # Encode variable embeddings with claim context
        var_claim_encoded = self.claim_graph_encoder(
            torch.cat([variable_embeddings, claim_context.unsqueeze(1).expand(-1, self.n_vars, -1)], dim=-1)
        )
        
        # Predict edges from data
        W_data_logits = self.predict_edges(var_data_encoded, dataset_embedding, self.data_edge_predictor)
        
        # Predict edges from claims
        W_claim_logits = self.predict_edges(var_claim_encoded, claim_context, self.claim_edge_predictor)
        
        # Compute gating: how much to trust data vs claims
        # Features: [n_samples_normalized, avg_claim_confidence, data_uncertainty]
        # Data uncertainty ≈ entropy of W_data
        W_data_probs = torch.sigmoid(W_data_logits)
        data_entropy = -(W_data_probs * torch.log(W_data_probs + 1e-8) + 
                         (1 - W_data_probs) * torch.log(1 - W_data_probs + 1e-8)).mean(dim=[1, 2])
        
        n_samples_normalized = torch.tensor([min(n_samples / 1000.0, 1.0)], device=device).expand(batch_size)
        claim_conf = torch.tensor([avg_claim_confidence], device=device).expand(batch_size)
        
        gate_features = torch.stack([n_samples_normalized, claim_conf, data_entropy], dim=-1)  # [batch, 3]
        gate = self.gate_network(gate_features)  # [batch, 1]
        
        # Combine data and claim evidence
        # gate ≈ 1 → trust data more
        # gate ≈ 0 → trust claims more
        gate = gate.unsqueeze(-1)  # [batch, 1, 1]
        W_posterior_logits = gate * W_data_logits + (1 - gate) * W_claim_logits
        
        # Compute acyclicity loss
        acyclicity_loss = torch.tensor(0.0, device=device)
        if self.use_notears_constraint:
            acyclicity_loss = self.compute_acyclicity_loss(W_posterior_logits)
        
        # Return components if requested
        components = None
        if return_components:
            components = {
                'W_data': torch.sigmoid(W_data_logits),
                'W_claim': torch.sigmoid(W_claim_logits),
                'gate': gate.squeeze(),
                'acyclicity_loss': acyclicity_loss,
                'W_posterior_logits': W_posterior_logits
            }
        
        # Convert to probabilities
        W_posterior = torch.sigmoid(W_posterior_logits)
        
        return W_posterior, components


if __name__ == "__main__":
    # Test the module
    batch_size = 4
    n_vars = 10
    d_model = 128
    
    # Create dummy inputs
    dataset_emb = torch.randn(batch_size, d_model)
    var_embs = torch.randn(batch_size, n_vars, d_model)
    claim_ctx = torch.randn(batch_size, d_model)
    
    # Create graph posterior head
    head = GraphPosteriorHead(n_vars=n_vars, d_model=d_model)
    
    # Forward pass
    W_posterior, components = head(
        dataset_emb,
        var_embs,
        claim_ctx,
        n_samples=500,
        avg_claim_confidence=0.8,
        return_components=True
    )
    
    print(f"Posterior adjacency shape: {W_posterior.shape}")  # [4, 10, 10]
    print(f"Acyclicity loss: {components['acyclicity_loss'].item():.4f}")
    print(f"Gate values (data trust): {components['gate'][0].item():.4f}")
    print("✓ Graph posterior head test passed!")
