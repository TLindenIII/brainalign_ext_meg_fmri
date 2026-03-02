import torch
import torch.nn.functional as F

def clip_loss(p_brain, y_clip, logit_scale):
    """
    Symmetric Contrastive Loss (InfoNCE) used by CLIP.
    
    Args:
        p_brain (Tensor): L2-normalized predicted brain embeddings (B, D)
        y_clip (Tensor): L2-normalized target CLIP embeddings (B, D)
        logit_scale (Tensor): Learnable scaling factor (temperature τ = 1/exp(logit_scale))
    
    Returns:
        Loss scalar
    """
    # Exponentiate the logit scale (keeps it strictly positive constraint)
    scale = logit_scale.exp()
    
    # Cosine Similarity matrix scaled by temperature
    # (B, D) @ (D, B) -> (B, B) matrix 
    logits = scale * p_brain @ y_clip.T
    
    # Ground truth: the diagonal elements are the matching pairs
    # Shape: (B,)
    batch_size = p_brain.size(0)
    labels = torch.arange(batch_size, device=p_brain.device)
    
    # Cross Entropy over both directions
    # L_b2i: Given a brain embedding, find the right image
    loss_b2i = F.cross_entropy(logits, labels)
    
    # L_i2b: Given an image embedding, find the right brain trial
    loss_i2b = F.cross_entropy(logits.T, labels)
    
    # Symmetric average
    return (loss_b2i + loss_i2b) / 2
