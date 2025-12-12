def dice_loss(pred, target, smooth=1e-6):
    # pred: [B, C, H, W]
    # target: [B, C, H, W]
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    # Calculate Dice coefficient
    dice_coeff = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice_coeff   # dice_loss = 1 - dice_coeff

class DiceLoss:
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
    def __call__(self, pred, target):
        target = target.float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice_coeff = (2 * intersection + self.epsilon) / (union + self.epsilon)  # Dice coefficient
        return 1 - dice_coeff   # dice_loss = 1 - dice_coeff