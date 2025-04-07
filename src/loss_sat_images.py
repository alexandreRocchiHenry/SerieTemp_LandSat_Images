import torch
import torch.nn.functional as F

##############################
# Fonctions pour Lovasz-Softmax
##############################

class_weights = torch.tensor([1.13, 0.78, 0.18, 11.43, 0.29, 1.0, 8.0], dtype=torch.float32)

def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / union
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def flatten_probas(probas, labels, ignore_index):
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore_index is None:
        return probas, labels
    valid = (labels != ignore_index)
    return probas[valid], labels[valid]

def lovasz_softmax_flat(probas, labels, classes='present', class_weights=None):
    if class_weights is not None:
        class_weights = class_weights.to(probas.device)
    if probas.numel() == 0:
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes == 'present' else classes
    weight_sum = 0.0
    for c in class_to_sum:
        fg = (labels == c).float()  # masque pour la classe c
        if fg.sum() == 0:
            continue
        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        loss_c = torch.dot(errors_sorted, grad)
        if class_weights is not None:
            loss_c = loss_c * class_weights[c]
            weight_sum += class_weights[c]
        else:
            weight_sum += 1.0
        losses.append(loss_c)
    if len(losses) == 0:
        return torch.tensor(0., device=probas.device)
    return sum(losses) / weight_sum


def lovasz_softmax_loss(probas, labels, classes='present', per_image=False, ignore_index=255, class_weights=None):
    if per_image:
        loss = torch.mean(torch.stack([
            lovasz_softmax_flat(*flatten_probas(probas[i:i+1], labels[i:i+1], ignore_index), classes=classes, class_weights=class_weights)
            for i in range(probas.size(0))
        ]))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore_index), classes=classes, class_weights=class_weights)
    return loss

def tversky_loss(pred_soft: torch.Tensor, target: torch.Tensor, num_classes: int,
                 alpha: float = 0.7, beta: float = 0.3, ignore_index: int = 255, smooth: float = 1e-6,
                 class_weights=None) -> torch.Tensor:
    if class_weights is not None:
        class_weights = class_weights.to(pred_soft.device)
    mask = (target != ignore_index)
    if mask.sum() == 0:
        return torch.tensor(0., device=pred_soft.device)
    
    target_clamped = target.clone()
    target_clamped[~mask] = 0
    one_hot = F.one_hot(target_clamped, num_classes).permute(0, 3, 1, 2).float()
    one_hot = one_hot * mask.unsqueeze(1)
    pred_soft = pred_soft * mask.unsqueeze(1)
    
    TP = (pred_soft * one_hot).sum(dim=(0, 2, 3))
    FP = (pred_soft * (1 - one_hot)).sum(dim=(0, 2, 3))
    FN = ((1 - pred_soft) * one_hot).sum(dim=(0, 2, 3))
    tversky_index = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    losses = 1.0 - tversky_index  # perte par classe
    if class_weights is not None:
        losses = losses * class_weights[:num_classes] 
        return losses.mean()
    else:
        return losses.mean()


def temporal_consistency_loss(preds_soft: torch.Tensor) -> torch.Tensor:
    T = preds_soft.shape[0]
    temp_loss = torch.tensor(0., device=preds_soft.device)
    if T > 1:
        for t in range(T - 1):
            log_p = torch.log(preds_soft[t+1] + 1e-8)
            log_q = torch.log(preds_soft[t] + 1e-8)
            kl_pq = F.kl_div(log_p, preds_soft[t], reduction='batchmean')
            kl_qp = F.kl_div(log_q, preds_soft[t+1], reduction='batchmean')
            temp_loss += (kl_pq + kl_qp) / 2
        temp_loss /= (T - 1)
    return temp_loss

def temporal_semi_supervised_loss(preds: torch.Tensor, sample: dict,
                                  lambda_temp: float = 1.0,
                                  lambda_tversky: float = 1.0,
                                  lambda_lovasz: float = 1.0,
                                  alpha_tversky: float = 0.7,
                                  beta_tversky: float = 0.3,
                                  num_classes: int = 8,
                                  class_weights=None) -> torch.Tensor:
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
            lovasz = lovasz_softmax_loss(p, y, ignore_index=255, class_weights=class_weights)
            p_soft = F.softmax(p, dim=1)
            tversky = tversky_loss(p_soft, y, num_classes, alpha=alpha_tversky, beta=beta_tversky, class_weights=class_weights)
            sup_loss += lambda_lovasz * lovasz + lambda_tversky * tversky
            valid_count += valid.sum()
    
    sup_loss = sup_loss / max(valid_count, 1)
    preds_soft = F.softmax(preds, dim=2)
    temp_loss = temporal_consistency_loss(preds_soft)
    
    return sup_loss + lambda_temp * temp_loss

##############################

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255, class_weights=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] (sorties du modèle, non-normalisées)
        targets: [B, H, W] (label avec index de classe)
        """
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index)
            if mask.sum() == 0:
                return torch.tensor(0., device=logits.device)
        else:
            mask = torch.ones_like(targets, dtype=torch.bool)

       
        if self.class_weights is not None:
            class_weights = self.class_weights.to(logits.device)
        else:
            class_weights = None

        loss = F.cross_entropy(logits, targets,
                               weight=class_weights,
                               ignore_index=self.ignore_index,
                               reduction='mean')
        return loss


class WeightedDiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, class_weights=None, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_weights = class_weights
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] (sorties du modèle)
        targets: [B, H, W] (label avec index de classe)
        """
        probs = F.softmax(logits, dim=1)

        mask = (targets != self.ignore_index)
        if mask.sum() == 0:
            return torch.tensor(0., device=logits.device)
        
        targets_clamped = targets.clone()
        targets_clamped[~mask] = 0
        
        one_hot = F.one_hot(targets_clamped, self.num_classes).permute(0, 3, 1, 2).float()
        # On applique le masque
        one_hot = one_hot * mask.unsqueeze(1)
        probs = probs * mask.unsqueeze(1)

        # Calcule Dice par classe
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        union = probs.sum(dims) + one_hot.sum(dims)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score  # on prend (1 - Dice)

        # Applique les poids de classe
        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)[:self.num_classes]
            dice_loss = dice_loss * w
        
        return dice_loss.mean()


import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    def __init__(self, ignore_index=255, class_weights=None, gamma=2.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.class_weights = class_weights
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W]
        targets: [B, H, W]
        """
        mask = (targets != self.ignore_index) if self.ignore_index is not None else None
        if mask is not None and mask.sum() == 0:
            return torch.tensor(0., device=logits.device)

        probs = F.softmax(logits, dim=1)

        pt = probs.gather(1, targets.unsqueeze(1))  # [B, 1, H, W]
        pt = pt.squeeze(1)                          # [B, H, W]


        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)
            class_w = w[targets]  # [B, H, W]
        else:
            class_w = 1.0

        if mask is not None:
            pt = pt[mask]
            class_w = class_w[mask]

        focal_loss = - class_w * (1 - pt).pow(self.gamma) * pt.log()

        return focal_loss.mean()


import torch
import torch.nn.functional as F

def temporal_semi_supervised_dice_loss(
    preds: torch.Tensor,
    sample: dict,
    dice_criterion,
    lambda_temp: float = 1.0
) -> torch.Tensor:
    """
    preds: [T, B, C, H, W] (sortie du réseau)
    sample: dictionnaire {"Y": [B, T, H, W], "mask_superv": [B, T]}
    dice_criterion: instance de WeightedDiceLoss
    lambda_temp: coefficient pour la consistance temporelle
    """
    if preds.shape[0] != sample["mask_superv"].shape[1]:
        preds = preds.permute(1, 0, 2, 3, 4)

    T, B, C, H, W = preds.shape
    Y = sample["Y"].permute(1, 0, 2, 3)             # [T, B, H, W]
    mask = sample["mask_superv"].permute(1, 0).bool()  # [T, B]

    sup_loss = torch.tensor(0., device=preds.device)
    valid_count = 0

    # ===== PERTE SUPERVISÉE (DICE) =====
    for t in range(T):
        valid = mask[t]  # booleen [B]
        if valid.any():
            logits_t = preds[t, valid]  # [n_valid, C, H, W]
            targets_t = Y[t, valid]     # [n_valid, H, W]
            loss_dice = dice_criterion(logits_t, targets_t)
            sup_loss += loss_dice
            valid_count += valid.sum()

    sup_loss = sup_loss / max(valid_count, 1)

    # ===== CONSISTANCE TEMPORELLE (facultative) =====
    preds_soft = F.softmax(preds, dim=2)  # [T, B, C, H, W]
    temp_loss = torch.tensor(0., device=preds.device)
    if T > 1:
        for t in range(T - 1):
            p_t   = preds_soft[t]
            p_t1  = preds_soft[t+1]
            log_p = torch.log(p_t + 1e-8)
            log_q = torch.log(p_t1 + 1e-8)
            kl_pq = F.kl_div(log_p, p_t1, reduction='batchmean')
            kl_qp = F.kl_div(log_q, p_t,  reduction='batchmean')
            temp_loss += (kl_pq + kl_qp) / 2
        temp_loss /= (T - 1)

    return sup_loss + lambda_temp * temp_loss
