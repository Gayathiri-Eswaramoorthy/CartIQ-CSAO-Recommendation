"""
Transformer-based ranking model.
Phase 4: Transformer architecture for sequential add-on recommendation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding for cart sequence."""
    
    def __init__(self, d_model: int, max_seq_len: int = 10):
        super().__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """Add positional encoding to embeddings."""
        return x + self.pe[:, :x.size(1), :]


class TransformerRecommender(nn.Module):
    """
    Transformer-based ranking model for sequential add-on recommendation.
    
    Architecture:
    - Item embedding (learned)
    - Positional encoding
    - Transformer encoder (2 layers, 4 heads)
    - Cart representation (mean pooling)
    - Multi-feature fusion
    - Binary classification head
    """
    
    def __init__(self, config: dict, dropout: float = 0.1):
        super().__init__()
        
        self.config = config
        self.embedding_dim = 128
        self.num_transformer_layers = 2
        self.num_heads = 4
        self.ff_dim = 256
        
        # Item embeddings (num_items + 1 to include padding token)
        self.padding_idx = config['num_items']  # Use num_items as padding index
        self.item_embedding = nn.Embedding(
            config['num_items'] + 1,  # +1 for padding token
            self.embedding_dim,
            padding_idx=self.padding_idx
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            self.embedding_dim,
            max_seq_len=config['max_cart_length']
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_transformer_layers
        )
        
        # Candidate embedding (same as item embedding)
        # Kept separate for clarity
        
        # Fusion network
        # Input dims:
        # - cart_embedding: embedding_dim
        # - candidate_embedding: embedding_dim
        # - user_features: user_feature_dim
        # - rest_features: rest_feature_dim
        # - context_features: context_feature_dim
        fusion_input_dim = (
            self.embedding_dim * 2 +  # cart + candidate
            config['user_feature_dim'] +
            config['rest_feature_dim'] +
            config['context_feature_dim']
        )
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        
    def forward(self, cart, candidate, user_features, rest_features, context_features):
        """
        Forward pass.
        
        Args:
            cart: (batch_size, max_cart_length) - item IDs with -1 padding
            candidate: (batch_size, 1) - candidate item ID
            user_features: (batch_size, user_feature_dim)
            rest_features: (batch_size, rest_feature_dim)
            context_features: (batch_size, context_feature_dim)
        
        Returns:
            scores: (batch_size, 1) - ranking score [0, 1]
        """
        batch_size = cart.size(0)
        
        # ==================
        # Cart Processing
        # ==================
        
        # Create attention mask for padding (item_id == -1) BEFORE replacement
        padding_mask = (cart == -1)  # (batch, max_len)
        
        # Replace -1 padding with valid padding index for embedding
        cart = cart.clone()  # Don't modify original tensor
        cart[cart == -1] = self.padding_idx
        
        # Embed cart items
        cart_embeddings = self.item_embedding(cart)  # (batch, max_len, embed_dim)
        
        # Add positional encoding
        cart_embeddings = self.positional_encoding(cart_embeddings)
        
        # Transformer encoder
        cart_transformer_out = self.transformer_encoder(
            cart_embeddings,
            src_key_padding_mask=padding_mask
        )  # (batch, max_len, embed_dim)
        
        # Apply layer norm
        cart_transformer_out = self.layer_norm(cart_transformer_out)
        
        # Mean pooling over valid positions (excluding padding)
        # Mask out padding positions before pooling
        mask_expanded = (~padding_mask).unsqueeze(-1).float()  # (batch, max_len, 1)
        cart_sum = (cart_transformer_out * mask_expanded).sum(dim=1)  # (batch, embed_dim)
        valid_count = mask_expanded.sum(dim=1)  # (batch, 1)
        valid_count = torch.clamp(valid_count, min=1.0)  # Avoid division by zero
        cart_embedding = cart_sum / valid_count  # (batch, embed_dim)
        
        # ==================
        # Candidate Processing
        # ==================
        
        candidate_embedding = self.item_embedding(candidate).squeeze(1)  # (batch, embed_dim)
        
        # ==================
        # Feature Fusion
        # ==================
        
        fused_input = torch.cat([
            cart_embedding,           # (batch, embedding_dim)
            candidate_embedding,      # (batch, embedding_dim)
            user_features,            # (batch, user_feature_dim)
            rest_features,            # (batch, rest_feature_dim)
            context_features          # (batch, context_feature_dim)
        ], dim=1)
        
        # Score
        score = self.fusion_mlp(fused_input)  # (batch, 1)
        
        return score


class TransformerRecommenderAblation(nn.Module):
    """
    Ablated versions of the Transformer model for ablation study.
    """
    
    def __init__(self, config: dict, ablation_type: str = "full", dropout: float = 0.1):
        """
        Types:
        - 'full': Full model (baseline)
        - 'no_sequence': Remove positional encoding, use mean pooling without order
        - 'no_user_features': Remove user features
        - 'no_context_features': Remove context features
        """
        super().__init__()
        
        self.config = config
        self.embedding_dim = 128
        self.num_transformer_layers = 2
        self.num_heads = 4
        self.ff_dim = 256
        self.ablation_type = ablation_type
        
        # Item embeddings (num_items + 1 to include padding token)
        self.padding_idx = config['num_items']  # Use num_items as padding index
        self.item_embedding = nn.Embedding(
            config['num_items'] + 1,  # +1 for padding token
            self.embedding_dim,
            padding_idx=self.padding_idx
        )
        
        # ==================
        # Architecture based on ablation type
        # ==================
        
        if ablation_type == "no_sequence":
            # No transformer, just mean pooling
            self.positional_encoding = None
            self.transformer_encoder = None
        else:
            # Standard transformer
            self.positional_encoding = PositionalEncoding(
                self.embedding_dim,
                max_seq_len=config['max_cart_length']
            )
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_dim,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.num_transformer_layers
            )
        
        # Fusion network
        fusion_input_dim = self.embedding_dim * 2  # cart + candidate
        
        if ablation_type != "no_user_features":
            fusion_input_dim += config['user_feature_dim']
        
        if ablation_type != "no_context_features":
            fusion_input_dim += config['context_feature_dim']
        
        fusion_input_dim += config['rest_feature_dim']  # Always include restaurant features
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        
    def forward(self, cart, candidate, user_features, rest_features, context_features):
        """Forward pass with ablations."""
        
        # Create padding mask BEFORE replacement
        padding_mask = (cart == -1)
        
        # Replace -1 padding with valid padding index for embedding
        cart = cart.clone()  # Don't modify original tensor
        cart[cart == -1] = self.padding_idx
        
        # Cart embeddings
        cart_embeddings = self.item_embedding(cart)
        
        # Process based on ablation type
        if self.ablation_type == "no_sequence":
            # No positional encoding or transformer
            # Simple mean pooling
            mask_expanded = (~padding_mask).unsqueeze(-1).float()
            cart_sum = (cart_embeddings * mask_expanded).sum(dim=1)
            valid_count = torch.clamp(mask_expanded.sum(dim=1), min=1.0)
            cart_embedding = cart_sum / valid_count
        else:
            # Standard transformer
            cart_embeddings = self.positional_encoding(cart_embeddings)
            cart_transformer_out = self.transformer_encoder(
                cart_embeddings,
                src_key_padding_mask=padding_mask
            )
            cart_transformer_out = self.layer_norm(cart_transformer_out)
            
            # Mean pooling
            mask_expanded = (~padding_mask).unsqueeze(-1).float()
            cart_sum = (cart_transformer_out * mask_expanded).sum(dim=1)
            valid_count = torch.clamp(mask_expanded.sum(dim=1), min=1.0)
            cart_embedding = cart_sum / valid_count
        
        # Candidate embedding
        candidate_embedding = self.item_embedding(candidate).squeeze(1)
        
        # Feature fusion
        fused_list = [
            cart_embedding,
            candidate_embedding,
            rest_features,
        ]
        
        if self.ablation_type != "no_user_features":
            fused_list.insert(2, user_features)
        
        if self.ablation_type != "no_context_features":
            fused_list.append(context_features)
        
        fused_input = torch.cat(fused_list, dim=1)
        score = self.fusion_mlp(fused_input)
        
        return score
