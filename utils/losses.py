"""
Loss functions for Graph-Enhanced EEG-to-Text model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeLoss(nn.Module):
    """
    Composite loss function combining:
    1. Cross-entropy loss for token prediction
    2. Graph smoothness regularization
    3. Contrastive alignment loss
    """
    
    def __init__(
        self,
        lambda_smooth: float = 0.1,
        lambda_contrastive: float = 0.2,
        vocab_size: int = 10000,
        ignore_index: int = -100
    ):
        """
        Args:
            lambda_smooth: Weight for graph smoothness loss
            lambda_contrastive: Weight for contrastive alignment loss
            vocab_size: Vocabulary size for cross-entropy
            ignore_index: Index to ignore in cross-entropy loss
        """
        super(CompositeLoss, self).__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_contrastive = lambda_contrastive
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def cross_entropy_loss(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Compute cross-entropy loss
        
        Args:
            logits: Model predictions (batch_size, seq_len, vocab_size)
            targets: Ground truth tokens (batch_size, seq_len)
            
        Returns:
            loss: Cross-entropy loss
        """
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        loss = self.ce_loss(logits_flat, targets_flat)
        return loss
    
    def graph_smoothness_loss(
        self,
        node_embeddings: torch.Tensor,
        adjacency_matrix: torch.Tensor
    ):
        """
        Graph smoothness regularization: encourages adjacent nodes to have similar embeddings
        
        Args:
            node_embeddings: Node embeddings (batch_size, num_nodes, embed_dim)
            adjacency_matrix: Adjacency matrix (batch_size, num_nodes, num_nodes)
            
        Returns:
            loss: Smoothness loss
        """
        batch_size, num_nodes, embed_dim = node_embeddings.shape
        
        # Compute pairwise differences
        h_i = node_embeddings.unsqueeze(2)  # (batch_size, num_nodes, 1, embed_dim)
        h_j = node_embeddings.unsqueeze(1)  # (batch_size, 1, num_nodes, embed_dim)
        
        diff = h_i - h_j  # (batch_size, num_nodes, num_nodes, embed_dim)
        diff_norm = torch.norm(diff, dim=-1) ** 2  # (batch_size, num_nodes, num_nodes)
        
        # Weight by adjacency
        A_expanded = adjacency_matrix.unsqueeze(0) if adjacency_matrix.dim() == 2 else adjacency_matrix
        loss = torch.sum(A_expanded * diff_norm, dim=(1, 2))  # (batch_size,)
        
        # Average over batch and normalize
        loss = loss.mean() / (num_nodes * num_nodes)
        
        return loss
    
    def contrastive_loss(
        self,
        eeg_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        temperature: float = 0.07
    ):
        """
        Contrastive alignment loss: aligns EEG and text embeddings
        
        Args:
            eeg_embeddings: EEG/STRE embeddings (batch_size, embed_dim)
            text_embeddings: Text embeddings (batch_size, embed_dim)
            temperature: Temperature scaling for contrastive loss
            
        Returns:
            loss: Contrastive loss
        """
        # Normalize embeddings
        eeg_norm = F.normalize(eeg_embeddings, p=2, dim=1)
        text_norm = F.normalize(text_embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(eeg_norm, text_norm.T) / temperature  # (batch_size, batch_size)
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(similarity.shape[0], device=similarity.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        node_embeddings: torch.Tensor = None,
        adjacency_matrix: torch.Tensor = None,
        eeg_embeddings: torch.Tensor = None,
        text_embeddings: torch.Tensor = None
    ):
        """
        Compute composite loss
        
        Args:
            logits: Decoder output logits (batch_size, seq_len, vocab_size)
            targets: Target tokens (batch_size, seq_len)
            node_embeddings: Node embeddings for smoothness loss (optional)
            adjacency_matrix: Adjacency matrix for smoothness loss (optional)
            eeg_embeddings: EEG embeddings for contrastive loss (optional)
            text_embeddings: Text embeddings for contrastive loss (optional)
            
        Returns:
            total_loss: Total loss
            loss_dict: Dictionary of individual loss components
        """
        # Cross-entropy loss (always computed)
        ce_loss = self.cross_entropy_loss(logits, targets)
        
        loss_dict = {'ce_loss': ce_loss.item()}
        total_loss = ce_loss
        
        # Graph smoothness loss
        if node_embeddings is not None and adjacency_matrix is not None:
            smooth_loss = self.graph_smoothness_loss(node_embeddings, adjacency_matrix)
            loss_dict['smooth_loss'] = smooth_loss.item()
            total_loss = total_loss + self.lambda_smooth * smooth_loss
        
        # Contrastive alignment loss
        if eeg_embeddings is not None and text_embeddings is not None:
            contrastive_loss = self.contrastive_loss(eeg_embeddings, text_embeddings)
            loss_dict['contrastive_loss'] = contrastive_loss.item()
            total_loss = total_loss + self.lambda_contrastive * contrastive_loss
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict

