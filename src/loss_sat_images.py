import torch
import torch.nn.functional as F

def dice_loss_frame(pred_soft, target, num_classes, ignore_index=255, smooth=1e-5):
    """
    Calcule la Dice Loss pour une frame donnée.
    
    Args:
      pred_soft (torch.Tensor): Probabilités prédictives [C, H, W] (après softmax).
      target (torch.Tensor): Masque ground truth [H, W] (classes 0-7 ou 255 pour ignorer).
      num_classes (int): Nombre total de classes (ex. 8).
      ignore_index (int): Valeur à ignorer dans le target.
      smooth (float): Terme de lissage.
      
    Returns:
      dice_loss (torch.Tensor): Dice Loss pour la frame.
    """
    valid_mask = (target != ignore_index).float()  # [H, W]
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_soft.device)
    # On remplace les pixels ignorés par 0 dans le target pour la conversion one-hot
    target_clean = target.clone().masked_fill(valid_mask == 0, 0)
    target_onehot = F.one_hot(target_clean, num_classes=num_classes)  # [H, W, C]
    target_onehot = target_onehot.permute(2, 0, 1).float()  # [C, H, W]
    # Appliquer le masque de validité
    pred_soft = pred_soft * valid_mask
    target_onehot = target_onehot * valid_mask
    dice_per_class = []
    for c in range(num_classes):
        p = pred_soft[c].view(-1)
        t = target_onehot[c].view(-1)
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        dice_c = (2.0 * intersection + smooth) / (union + smooth)
        dice_per_class.append(dice_c)
    dice_frame = 1 - torch.mean(torch.stack(dice_per_class))
    return dice_frame

def focal_loss_frame(pred, target, alpha=0.25, gamma=2.0, ignore_index=255):
    """
    Calcule la Focal Loss pour une frame donnée.
    
    Args:
      pred (torch.Tensor): Logits prédits [C, H, W].
      target (torch.Tensor): Masque ground truth [H, W] (classes 0-7 ou 255 pour ignorer).
      alpha (float): Facteur de pondération.
      gamma (float): Exposant pour diminuer l'effet des exemples faciles.
      ignore_index (int): Valeur à ignorer dans le target.
      
    Returns:
      focal_loss (torch.Tensor): Focal Loss moyenne sur les pixels valides.
    """
    valid_mask = (target != ignore_index)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    # On ajoute une dimension batch fictive
    pred = pred.unsqueeze(0)       # [1, C, H, W]
    target = target.unsqueeze(0)   # [1, H, W]
    logpt = F.log_softmax(pred, dim=1)
    pt = torch.exp(logpt)
    # Récupérer la probabilité associée à la classe vraie
    target_unsqueezed = target.unsqueeze(1)  # [1, 1, H, W]
    logpt = logpt.gather(1, target_unsqueezed)  # [1, 1, H, W]
    pt = pt.gather(1, target_unsqueezed)        # [1, 1, H, W]
    valid_mask = valid_mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
    loss = -alpha * (1 - pt) ** gamma * logpt * valid_mask
    return loss.sum() / valid_mask.sum()

def temporal_semi_supervised_loss(preds, sample, lambda_temp=1.0, lambda_dice=1.0, lambda_focal=0.0, num_classes=8):
    """
    Accepts preds of shape [B, C, H, W] or [T, C, H, W].
    Converts single-frame outputs into time-first shape automatically.
    """
    # If model returned [B, C, H, W], add a time dimension
    if preds.ndim == 4:
        preds = preds.unsqueeze(0)  # → [1, C, H, W]
        sample = {
            "Y": sample["Y"].unsqueeze(0),
            "mask_superv": sample["mask_superv"].unsqueeze(0)
        }

    T, C, H, W = preds.shape
    Y = sample["Y"]
    mask_superv = sample["mask_superv"]

    supervised_ce = supervised_dice = supervised_focal = 0.0
    annotated_frames = 0

    for t in range(T):
        if mask_superv[t]:
            supervised_ce += F.cross_entropy(preds[t].unsqueeze(0), Y[t].unsqueeze(0), ignore_index=255)
            pred_soft = torch.softmax(preds[t], dim=0)
            supervised_dice += dice_loss_frame(pred_soft, Y[t], num_classes, ignore_index=255)
            supervised_focal += focal_loss_frame(preds[t], Y[t], ignore_index=255)
            annotated_frames += 1

    if annotated_frames > 0:
        supervised_loss = (supervised_ce/annotated_frames) \
                         + lambda_dice*(supervised_dice/annotated_frames) \
                         + lambda_focal*(supervised_focal/annotated_frames)
    else:
        supervised_loss = torch.tensor(0.0, device=preds.device)

    preds_soft = torch.softmax(preds, dim=1)
    temporal_loss = sum(torch.mean((preds_soft[t+1] - preds_soft[t])**2) for t in range(T-1))
    temporal_loss = temporal_loss/(T-1) if T>1 else torch.tensor(0.0, device=preds.device)

    return supervised_loss + lambda_temp*temporal_loss
