import torch
import torch.nn.functional as F

def dice_loss(pred_soft: torch.Tensor, target: torch.Tensor, num_classes: int,
              ignore_index: int = 255, smooth: float = 1e-6) -> torch.Tensor:
    mask = (target != ignore_index)
    if mask.sum() == 0:
        return torch.tensor(0., device=pred_soft.device)

    target_clamped = target.clone()
    target_clamped[~mask] = 0

    one_hot = F.one_hot(target_clamped, num_classes).permute(0, 3, 1, 2).float()
    one_hot = one_hot * mask.unsqueeze(1)
    pred_soft = pred_soft * mask.unsqueeze(1)

    intersection = (pred_soft * one_hot).sum(dim=(0, 2, 3))
    union = pred_soft.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    valid = union > 0

    if valid.any():
        dice_score = ((2 * intersection[valid] + smooth) / (union[valid] + smooth)).mean()
    else:
        dice_score = torch.tensor(1.0, device=pred_soft.device)

    return 1.0 - dice_score

def focal_loss(pred: torch.Tensor, target: torch.Tensor,
               alpha: float = 0.25, gamma: float = 2.0,
               ignore_index: int = 255) -> torch.Tensor:
    mask = (target != ignore_index)
    if mask.sum() == 0:
        return torch.tensor(0., device=pred.device)

    max_label = target[mask].max().item()
    assert max_label < pred.shape[1], f"Invalid target {max_label} >= {pred.shape[1]}"

    ce = F.cross_entropy(pred, target, ignore_index=ignore_index, reduction='none')
    pt = torch.exp(-ce)
    loss = alpha * (1 - pt) ** gamma * ce
    return (loss * mask.float()).sum() / mask.sum()

def temporal_semi_supervised_loss(preds: torch.Tensor, sample: dict,
                                  lambda_temp: float = 1.0,
                                  lambda_dice: float = 1.0,
                                  lambda_focal: float = 0.0,
                                  num_classes: int = 8) -> torch.Tensor:
    # preds: [T, B, C, H, W] or [B, T, C, H, W]
    if preds.shape[0] != sample["mask_superv"].shape[1]:
        preds = preds.permute(1, 0, 2, 3, 4)

    T, B, C, H, W = preds.shape
    Y = sample["Y"].permute(1, 0, 2, 3)           # [T, B, H, W]
    mask = sample["mask_superv"].permute(1, 0).bool()  # [T, B]

    sup_loss = torch.tensor(0., device=preds.device)
    valid_count = 0

    for t in range(T):
        valid = mask[t]
        if valid.any():
            p = preds[t, valid]  # [n_valid, C, H, W]
            y = Y[t, valid]      # [n_valid, H, W]
            sup_loss += F.cross_entropy(p, y, ignore_index=255)
            p_soft = F.softmax(p, dim=1)
            sup_loss += lambda_dice * dice_loss(p_soft, y, num_classes)
            sup_loss += lambda_focal * focal_loss(p, y)
            valid_count += valid.sum()

    sup_loss = sup_loss / max(valid_count, 1)

    temp_loss = torch.tensor(0., device=preds.device)
    if T > 1:
        preds_soft = F.softmax(preds, dim=2)
        for t in range(T - 1):
            temp_loss += F.smooth_l1_loss(preds_soft[t + 1], preds_soft[t])
        temp_loss /= (T - 1)

    return sup_loss + lambda_temp * temp_loss
