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
