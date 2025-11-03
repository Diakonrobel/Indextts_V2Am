"""Vocabulary utilities for proper embedding transfer"""
import torch
import torch.nn as nn
import sentencepiece as spm
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def load_vocab_mapping(vocab_file: str) -> Dict[str, int]:
    """Load vocabulary and create string-to-id mapping
    
    Args:
        vocab_file: Path to SentencePiece model file
        
    Returns:
        Dictionary mapping token strings to IDs
    """
    sp = spm.SentencePieceProcessor()
    sp.Load(vocab_file)
    
    vocab_map = {}
    for idx in range(sp.GetPieceSize()):
        token = sp.IdToPiece(idx)
        vocab_map[token] = idx
    
    return vocab_map


def resize_token_embeddings(
    old_embeddings: nn.Embedding,
    new_vocab_size: int,
    old_vocab_file: str,
    new_vocab_file: str,
    init_std: float = 0.02
) -> nn.Embedding:
    """Resize token embeddings with proper token-string mapping
    
    Args:
        old_embeddings: Original embedding layer
        new_vocab_size: Size of new vocabulary
        old_vocab_file: Path to old SentencePiece vocab
        new_vocab_file: Path to new SentencePiece vocab
        init_std: Standard deviation for new token initialization
        
    Returns:
        New embedding layer with transferred weights
    """
    embedding_dim = old_embeddings.embedding_dim
    old_vocab_size = old_embeddings.num_embeddings
    
    # Load vocabulary mappings
    try:
        old_vocab_map = load_vocab_mapping(old_vocab_file)
        new_vocab_map = load_vocab_mapping(new_vocab_file)
    except Exception as e:
        logger.warning(f"Could not load vocab mappings: {e}. Using position-based fallback.")
        # Fallback to position-based copying
        new_embeddings = nn.Embedding(new_vocab_size, embedding_dim)
        min_size = min(old_vocab_size, new_vocab_size)
        with torch.no_grad():
            new_embeddings.weight[:min_size] = old_embeddings.weight[:min_size]
            if new_vocab_size > old_vocab_size:
                new_embeddings.weight[old_vocab_size:].normal_(mean=0.0, std=init_std)
        return new_embeddings
    
    # Create new embedding layer
    new_embeddings = nn.Embedding(new_vocab_size, embedding_dim)
    
    # Initialize with proper distribution
    nn.init.normal_(new_embeddings.weight, mean=0.0, std=init_std)
    
    # Copy embeddings for matching tokens
    transferred_count = 0
    new_token_count = 0
    
    with torch.no_grad():
        for new_token, new_id in new_vocab_map.items():
            if new_token in old_vocab_map:
                # Token exists in old vocab - copy its embedding
                old_id = old_vocab_map[new_token]
                new_embeddings.weight[new_id] = old_embeddings.weight[old_id]
                transferred_count += 1
            else:
                # New token - keep random initialization
                new_token_count += 1
    
    logger.info(f"Vocabulary resizing complete:")
    logger.info(f"  Old vocab size: {old_vocab_size}")
    logger.info(f"  New vocab size: {new_vocab_size}")
    logger.info(f"  Transferred embeddings: {transferred_count}")
    logger.info(f"  New tokens (random init): {new_token_count}")
    logger.info(f"  Transfer ratio: {transferred_count/new_vocab_size:.1%}")
    
    return new_embeddings


def resize_linear_layer(
    old_linear: nn.Linear,
    new_vocab_size: int,
    old_vocab_file: str,
    new_vocab_file: str,
    init_std: float = 0.02
) -> nn.Linear:
    """Resize linear output layer (e.g., text_head) with proper token mapping
    
    Args:
        old_linear: Original linear layer
        new_vocab_size: Size of new vocabulary
        old_vocab_file: Path to old SentencePiece vocab
        new_vocab_file: Path to new SentencePiece vocab
        init_std: Standard deviation for initialization
        
    Returns:
        New linear layer with transferred weights
    """
    hidden_size = old_linear.in_features
    old_vocab_size = old_linear.out_features
    
    # Load vocabulary mappings
    try:
        old_vocab_map = load_vocab_mapping(old_vocab_file)
        new_vocab_map = load_vocab_mapping(new_vocab_file)
    except Exception as e:
        logger.warning(f"Could not load vocab mappings for linear layer: {e}")
        # Fallback
        new_linear = nn.Linear(hidden_size, new_vocab_size)
        min_size = min(old_vocab_size, new_vocab_size)
        with torch.no_grad():
            new_linear.weight[:min_size] = old_linear.weight[:min_size]
            if new_linear.bias is not None and old_linear.bias is not None:
                new_linear.bias[:min_size] = old_linear.bias[:min_size]
            if new_vocab_size > old_vocab_size:
                new_linear.weight[old_vocab_size:].normal_(mean=0.0, std=init_std)
        return new_linear
    
    # Create new linear layer
    new_linear = nn.Linear(hidden_size, new_vocab_size, bias=old_linear.bias is not None)
    
    # Initialize
    nn.init.normal_(new_linear.weight, mean=0.0, std=init_std)
    if new_linear.bias is not None:
        nn.init.zeros_(new_linear.bias)
    
    # Copy weights for matching tokens
    transferred_count = 0
    
    with torch.no_grad():
        for new_token, new_id in new_vocab_map.items():
            if new_token in old_vocab_map:
                old_id = old_vocab_map[new_token]
                new_linear.weight[new_id] = old_linear.weight[old_id]
                if new_linear.bias is not None and old_linear.bias is not None:
                    new_linear.bias[new_id] = old_linear.bias[old_id]
                transferred_count += 1
    
    logger.info(f"Linear layer resizing: transferred {transferred_count}/{new_vocab_size} weights")
    
    return new_linear
