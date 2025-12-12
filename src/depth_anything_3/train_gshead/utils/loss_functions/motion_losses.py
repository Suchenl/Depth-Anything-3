import torch
import math

class MixtureOfLaplaceLoss:
    def __init__(self,
                 gamma=0.85,
                 use_var=True,
                 var_min=0,
                 var_max=10,
                 epsilon=1e-6):
        self.gamma = gamma
        self.use_var = use_var
        self.var_min = var_min
        self.var_max = var_max
        self.epsilon = epsilon

    def __call__(self, pred_motions, pred_infos, true_motion, valid_mask):
        return mixture_of_laplace_loss(
            pred_motions, pred_infos, true_motion, valid_mask,
            gamma=self.gamma,
            use_var=self.use_var, 
            var_min=0,
            var_max=10, 
            epsilon=1e-6
            )
                                       
def mixture_of_laplace_loss(pred_motions, pred_infos, true_motion, 
                            valid_mask=None,
                            gamma=0.85, 
                            use_var=True, 
                            var_min=0, 
                            var_max=10, 
                            epsilon=1e-6):
    device, dtype = true_motion.device, true_motion.dtype
    B, _, H, W = true_motion.shape
    
    # mixture of laplace loss in each scale
    num_preds = len(pred_motions)
    total_loss = 0.0

    if not use_var:
        var_max = var_min = 0

    if valid_mask is None:
        valid_mask = torch.ones((B, 1, H, W), device=device, dtype=dtype)
        
    for i in range(len(pred_infos)):
        raw_b = pred_infos[i][:, 2:]
        log_b = torch.zeros_like(raw_b)
        weight = pred_infos[i][:, :2]
        # Large b Component                
        log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
        # Small b Component
        log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
        # term2: [N, 2, m, H, W]
        term2 = ((true_motion - pred_motions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
        # term1: [N, m, H, W]
        term1 = weight - math.log(2) - log_b
        nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
        # Add to total loss
        weight = gamma ** (num_preds - i - 1)
        final_mask = (~torch.isnan(nf_loss.detach())) & (~torch.isinf(nf_loss.detach())) & valid_mask
        total_loss += weight * ((final_mask * nf_loss).sum() / (final_mask.sum() + epsilon))
    return total_loss