class AsymmetricLabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing only on clot (1) and wall (2).
    Blood (0) gets no smoothing.
    """
    def __init__(self, smoothing=0.1, weight=None, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.n_classes = 3

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        
        with torch.no_grad():
            # Start with one-hot
            target_onehot = torch.zeros_like(log_probs)
            target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
            
            # Apply smoothing ONLY to clot and wall
            smoothed = target_onehot.clone()
            
            # For clot (1) and wall (2)
            non_blood_mask = (target == 1) | (target == 2)
            if non_blood_mask.any():
                smoothed[non_blood_mask] = (1.0 - self.smoothing) * target_onehot[non_blood_mask] + \
                                           (self.smoothing / self.n_classes)

            # Optional: apply class weights
            if self.weight is not None:
                smoothed = smoothed * self.weight.unsqueeze(0)
                smoothed = smoothed / smoothed.sum(dim=1, keepdim=True)

        loss = - (smoothed * log_probs).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
            
criterion = AsymmetricLabelSmoothingCrossEntropy(
    smoothing=0.1,                    # tune this (0.08 ~ 0.15)
    weight=class_weights.to(DEVICE),
    reduction='mean'
)            