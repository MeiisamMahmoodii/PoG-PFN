"""
Module A: PFN-Style Dataset Encoder

Transforms tabular datasets into embeddings using a transformer architecture
similar to TabPFN. Produces both:
- A global dataset embedding E_D
- Per-variable embeddings e_i
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for row positions."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch, d_model]
        """
        return x + self.pe[:x.size(0)]


class DatasetEncoder(nn.Module):
    """
    TabPFN-style encoder that processes datasets as sequences of rows.
    
    Each row is tokenized as: [x1, x2, ..., xn, y] (features + target)
    The transformer processes these rows to produce:
    - Dataset embedding E_D: Global context representation
    - Variable embeddings e_i: Per-variable representations
    """
    
    def __init__(
        self,
        n_vars: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 500,
        use_positional_encoding: bool = True
    ):
        super().__init__()
        
        self.n_vars = n_vars
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Variable embedding: maps each variable index to an embedding
        self.variable_embedding = nn.Embedding(n_vars, d_model)
        
        # Value projection: projects scalar values to d_model
        self.value_projection = nn.Linear(1, d_model)
        
        # Row encoder: combines variable embedding + value
        self.row_encoder = nn.Linear(d_model * 2, d_model)
        
        # Positional encoding for row positions
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model, max_seq_len * n_vars)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projections
        self.dataset_proj = nn.Linear(d_model, d_model)
        self.variable_proj = nn.Linear(d_model, d_model)
        
    def tokenize_dataset(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize dataset into sequence of (variable_id, value) pairs.
        
        Args:
            X: Feature matrix [batch, n_samples, n_features]
            y: Target vector [batch, n_samples, 1] (optional)
            
        Returns:
            tokens: [batch, seq_len, d_model] where seq_len = n_samples * n_vars
            mask: [batch, seq_len] attention mask
        """
        batch_size, n_samples, n_features = X.shape
        device = X.device
        
        # Concatenate features and target if provided
        if y is not None:
            data = torch.cat([X, y.unsqueeze(-1)], dim=-1)  # [batch, n_samples, n_vars]
        else:
            data = X
            
        # Robust input clamping: handle extreme outliers that could cause Transformer NaNs
        data = torch.clamp(data, min=-1e3, max=1e3)
        
        n_vars = data.shape[-1]
        
        # Create variable indices [0, 1, ..., n_vars-1] repeated for each sample
        var_indices = torch.arange(n_vars, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, n_vars]
        var_indices = var_indices.expand(batch_size, n_samples, -1)  # [batch, n_samples, n_vars]
        
        # Get variable embeddings
        var_embeds = self.variable_embedding(var_indices)  # [batch, n_samples, n_vars, d_model]
        
        # Project values
        values = data.unsqueeze(-1)  # [batch, n_samples, n_vars, 1]
        value_embeds = self.value_projection(values)  # [batch, n_samples, n_vars, d_model]
        
        # Combine variable embedding + value embedding
        combined = torch.cat([var_embeds, value_embeds], dim=-1)  # [batch, n_samples, n_vars, 2*d_model]
        tokens = self.row_encoder(combined)  # [batch, n_samples, n_vars, d_model]
        
        # Flatten to sequence: [batch, seq_len, d_model] where seq_len = n_samples * n_vars
        tokens = tokens.reshape(batch_size, n_samples * n_vars, self.d_model)
        
        # Create attention mask (all valid for now)
        mask = torch.zeros(batch_size, n_samples * n_vars, dtype=torch.bool, device=device)
        
        return tokens, mask
    
    def forward(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode dataset into embeddings.
        
        Args:
            X: Feature matrix [batch, n_samples, n_features]
            y: Target vector [batch, n_samples] (optional)
            mask: Attention mask [batch, seq_len] (optional)
            
        Returns:
            dataset_embedding: Global dataset representation [batch, d_model]
            variable_embeddings: Per-variable representations [batch, n_vars, d_model]
        """
        # Tokenize dataset
        tokens, auto_mask = self.tokenize_dataset(X, y)
        if mask is None:
            mask = auto_mask
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            # Reshape for positional encoder (expects [seq, batch, dim])
            tokens = tokens.transpose(0, 1)
            tokens = self.pos_encoder(tokens)
            tokens = tokens.transpose(0, 1)
        
        # Pass through transformer
        # Note: PyTorch transformer expects src_key_padding_mask with True = masked
        encoded = self.transformer(tokens, src_key_padding_mask=mask)  # [batch, seq_len, d_model]
        
        # Extract dataset embedding (use mean pooling over all tokens)
        dataset_embedding = encoded.mean(dim=1)  # [batch, d_model]
        dataset_embedding = self.dataset_proj(dataset_embedding)
        
        # Extract variable embeddings
        # Group tokens by variable and pool
        batch_size, seq_len, _ = encoded.shape
        n_samples = seq_len // self.n_vars
        
        # Reshape to [batch, n_samples, n_vars, d_model]
        encoded_vars = encoded.reshape(batch_size, n_samples, self.n_vars, self.d_model)
        
        # Pool over samples to get per-variable embeddings
        variable_embeddings = encoded_vars.mean(dim=1)  # [batch, n_vars, d_model]
        variable_embeddings = self.variable_proj(variable_embeddings)
        
        return dataset_embedding, variable_embeddings


if __name__ == "__main__":
    # Test the module
    batch_size = 4
    n_samples = 100
    n_features = 10
    
    # Create dummy data
    X = torch.randn(batch_size, n_samples, n_features)
    y = torch.randn(batch_size, n_samples)
    
    # Create encoder
    encoder = DatasetEncoder(n_vars=n_features + 1, d_model=128, n_heads=4, n_layers=3)
    
    # Forward pass
    dataset_emb, var_embs = encoder(X, y)
    
    print(f"Dataset embedding shape: {dataset_emb.shape}")  # [4, 128]
    print(f"Variable embeddings shape: {var_embs.shape}")  # [4, 11, 128]
    print("✓ Dataset encoder test passed!")
