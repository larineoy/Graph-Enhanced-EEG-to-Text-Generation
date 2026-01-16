"""
Loss functions for Graph-Enhanced EEG-to-Text model
FIXED VERSION: Removed label smoothing that was causing comma-only predictions
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
        lambda_diversity: float = 1.0,  # NEW: Diversity regularization to prevent STRE collapse
        vocab_size: int = 10000,
        ignore_index: int = -100
    ):
        """
        Args:
            lambda_smooth: Weight for graph smoothness loss
            lambda_contrastive: Weight for contrastive alignment loss
            lambda_diversity: Weight for STRE embedding diversity loss (prevents collapse)
            vocab_size: Vocabulary size for cross-entropy
            ignore_index: Index to ignore in cross-entropy loss
        """
        super(CompositeLoss, self).__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_contrastive = lambda_contrastive
        self.lambda_diversity = lambda_diversity
        self.ignore_index = ignore_index
        
        # ========================================================================
        # CRITICAL FIX: Removed label_smoothing=0.1 that was causing collapse
        # ========================================================================
        # Label smoothing was causing the model to predict only commas (token 1010)
        # because it:
        # 1. Reduces confidence in correct predictions (90% instead of 100%)
        # 2. Distributes 10% probability to ALL tokens including wrong ones
        # 3. Combined with class imbalance (commas are very frequent), the model
        #    learns that predicting comma is "safer" than learning the actual task
        #
        # Solution: Disable label smoothing initially. Only add it after epoch 50+
        #           when the model has learned basic text generation.
        # ========================================================================
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=0.0)
    
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
        # Use reshape instead of view to handle non-contiguous tensors (e.g., from slicing)
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
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
        # Paper formula: L_smooth = sum_{i,j} A_ij ||h_i - h_j||^2
        loss = torch.sum(A_expanded * diff_norm, dim=(1, 2))  # (batch_size,)
        
        # Average over batch (no normalization per paper formula)
        loss = loss.mean()
        
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
        
        # Diversity loss: Prevent STRE embedding collapse
        # Encourages embeddings to be diverse by penalizing high similarity
        if eeg_embeddings is not None and eeg_embeddings.shape[0] > 1:
            diversity_loss = self.diversity_loss(eeg_embeddings)
            loss_dict['diversity_loss'] = diversity_loss.item()
            total_loss = total_loss + self.lambda_diversity * diversity_loss
    
            # NEW: Variance regularization
            batch_variance = torch.var(eeg_embeddings, dim=0).mean()
            variance_penalty = 1.0 / (batch_variance + 1e-6)
            loss_dict['variance_penalty'] = variance_penalty.item()
            loss_dict['batch_variance'] = batch_variance.item()
            total_loss = total_loss + 0.1 * variance_penalty
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def diversity_loss(self, embeddings: torch.Tensor):
        """
        Diversity loss to prevent embedding collapse.
        Penalizes high pairwise similarity between embeddings.
        
        Args:
            embeddings: STRE embeddings (batch_size, embed_dim)
            
        Returns:
            loss: Diversity loss (lower = more diverse)
        """
        batch_size = embeddings.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)  # (batch_size, embed_dim)
        
        # Compute pairwise cosine similarities
        # similarity[i,j] = cosine(emb[i], emb[j])
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())  # (batch_size, batch_size)
        
        # Remove diagonal (self-similarity = 1.0)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        off_diagonal_similarities = similarity_matrix[mask]
        
        # Penalize high similarities (encourage diversity)
        # Loss = mean of squared similarities (encourages them to be close to 0)
        diversity_loss = torch.mean(off_diagonal_similarities ** 2)
        
        return diversity_loss


# ============================================================================
# OPTIONAL: FocalLoss for handling class imbalance (e.g., too many commas)
# ============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss: Down-weights easy examples, up-weights hard examples.
    
    Helps with class imbalance by reducing the contribution of frequent tokens
    (like commas) and focusing learning on harder, less frequent tokens.
    
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    When gamma=0, this reduces to standard cross-entropy.
    When gamma>0, easy examples (p_t close to 1) are down-weighted.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = -100):
        """
        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples
            gamma: Focusing parameter (gamma >= 0). Higher = more focus on hard examples
            ignore_index: Index to ignore (e.g., padding token)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits: Predictions (batch*seq, vocab_size)
            targets: Ground truth (batch*seq)
        Returns:
            loss: Focal loss scalar
        """
        # Standard cross-entropy (no reduction yet)
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        
        # Get predicted probabilities for correct class
        p_t = torch.exp(-ce_loss)
        
        # Focal weight: (1 - p_t)^gamma
        # When p_t is close to 1 (easy example), weight is close to 0
        # When p_t is close to 0 (hard example), weight is close to 1
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weight and alpha
        focal_loss = self.alpha * focal_weight * ce_loss
        
        # Only average over non-ignored tokens
        mask = (targets != self.ignore_index)
        if mask.sum() > 0:
            return focal_loss[mask].mean()
        else:
            return focal_loss.mean()


# ============================================================================
# OPTIONAL: CompositeLoss with FocalLoss instead of CrossEntropyLoss
# ============================================================================
class CompositeLossWithFocal(CompositeLoss):
    """
    Same as CompositeLoss but uses FocalLoss instead of CrossEntropyLoss.
    
    Use this if your model is still predicting only frequent tokens after
    removing label smoothing.
    """
    
    def __init__(
        self,
        lambda_smooth: float = 0.1,
        lambda_contrastive: float = 0.2,
        vocab_size: int = 10000,
        ignore_index: int = -100,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__(lambda_smooth, lambda_contrastive, vocab_size, ignore_index)
        
        # Replace CrossEntropyLoss with FocalLoss
        self.ce_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, ignore_index=ignore_index)
        print(f"[CompositeLossWithFocal] Using FocalLoss (alpha={focal_alpha}, gamma={focal_gamma})")
