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
    Calcule la Focal Loss pour une frame donnée en utilisant la cross-entropy avec ignore_index.
    
    Args:
      pred (torch.Tensor): Logits prédits [C, H, W].
      target (torch.Tensor): Masque ground truth [H, W] (classes 0-7, 255 pour ignorer).
      alpha (float): Facteur de pondération.
      gamma (float): Exposant pour diminuer l'effet des exemples faciles.
      ignore_index (int): Valeur à ignorer dans le target.
      
    Returns:
      focal_loss (torch.Tensor): Focal Loss moyenne sur les pixels valides.
    """
    # On calcule la cross-entropy par pixel sans réduction
    ce_loss = F.cross_entropy(pred.unsqueeze(0), target.unsqueeze(0), reduction='none', ignore_index=ignore_index)[0]
    # On en déduit p_t (la probabilité de la classe vraie) : ce_loss = -log(p_t)  =>  p_t = exp(-ce_loss)
    pt = torch.exp(-ce_loss)
    # On applique la modulation focal : loss = alpha*(1-p_t)^gamma * ce_loss
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    # On crée un masque valide pour ignorer les pixels avec target==ignore_index
    valid_mask = (target != ignore_index).float()
    # On ne tient compte que des pixels valides
    focal_loss = focal_loss * valid_mask
    return focal_loss.sum() / valid_mask.sum()


def temporal_semi_supervised_loss(preds, sample, lambda_temp=1.0, lambda_dice=1.0, lambda_focal=1.0, num_classes=8):
    mask_superv = sample["mask_superv"]  # [B, T]
    Y = sample["Y"]                      # [B, T, H, W]
    
    # Si preds est de forme [B, T, C, H, W], on la transforme en [T, B, C, H, W]
    if preds.shape[0] != mask_superv.shape[1]:
        preds = preds.permute(1, 0, 2, 3, 4)
    
    T, B, C, H, W = preds.shape

    supervised_ce = supervised_dice = supervised_focal = 0.0
    annotated_frames = 0

    for t in range(T):
        valid = mask_superv[:, t]  # [B]
        if valid.any():
            preds_t = preds[t][valid]  # [n_valid, C, H, W]
            Y_t = Y[valid, t]          # [n_valid, H, W]

            supervised_ce += torch.nn.functional.cross_entropy(preds_t, Y_t, ignore_index=255)
            pred_soft = torch.softmax(preds_t, dim=1)
            for i in range(preds_t.size(0)):
                supervised_dice += dice_loss_frame(pred_soft[i], Y_t[i], num_classes)
                supervised_focal += focal_loss_frame(preds_t[i], Y_t[i])
            annotated_frames += valid.sum().item()

    supervised_loss = (supervised_ce + lambda_dice * supervised_dice + lambda_focal * supervised_focal) / max(annotated_frames, 1)

    preds_soft = torch.softmax(preds, dim=2)
    temporal_loss = torch.tensor(0.0, device=preds.device)
    if T > 1:
        temporal_loss = sum(((preds_soft[t+1] - preds_soft[t])**2).mean() for t in range(T-1)) / (T-1)

    return supervised_loss + lambda_temp * temporal_loss