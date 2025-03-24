import numpy as np
import torch

class MetricsEvaluator:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        # Pour les métriques statiques
        self.total_correct = 0
        self.total_valid = 0
        # Matrice de confusion : lignes = ground truth, colonnes = prédictions
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        # Pour la stabilité temporelle (moyenne de la différence entre frames consécutives)
        self.temporal_diff_sum = 0.0
        self.temporal_frames = 0

    def update(self, pred, target):
        """
        Met à jour les compteurs à partir d'un batch ou d'une frame.
        Args:
            pred (torch.Tensor or np.array): masque prédit, shape (B, H, W) ou (H, W)
            target (torch.Tensor or np.array): masque ground truth, shape identique, avec valeurs dans [0, num_classes-1] ou ignore_index
        """
        # Si on a un batch
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        if pred.ndim == 3:
            # Batch de frames
            B = pred.shape[0]
            for b in range(B):
                self._update_frame(pred[b], target[b])
        else:
            self._update_frame(pred, target)

    def _update_frame(self, pred, target):
        # On ne considère que les pixels valides (target != ignore_index)
        valid_mask = (target != self.ignore_index)
        if valid_mask.sum() == 0:
            return  # Aucun pixel valide dans cette frame
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        self.total_correct += (pred_valid == target_valid).sum()
        self.total_valid += valid_mask.sum()
        # Mise à jour de la matrice de confusion
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.confusion_matrix[i, j] += np.logical_and(target_valid == i, pred_valid == j).sum()
    
    def update_temporal(self, pred_seq):
        """
        Met à jour la métrique de stabilité temporelle à partir d'une séquence de prédictions.
        Args:
            pred_seq (torch.Tensor or np.array): séquence prédite, shape (T, H, W)
        """
        if isinstance(pred_seq, torch.Tensor):
            pred_seq = pred_seq.cpu().numpy()
        T = pred_seq.shape[0]
        if T < 2:
            return
        diff = 0.0
        count = 0
        for t in range(T - 1):
            # On calcule la différence moyenne absolue entre deux frames successives
            diff += np.mean(np.abs(pred_seq[t+1] - pred_seq[t]))
            count += 1
        self.temporal_diff_sum += diff
        self.temporal_frames += count

    def compute(self):
        # Accuracy pixel (sur pixels valides)
        pixel_accuracy = self.total_correct / self.total_valid if self.total_valid > 0 else 0.0

        # Calcul de l'IoU par classe
        ious = []
        for i in range(self.num_classes):
            intersection = self.confusion_matrix[i, i]
            union = self.confusion_matrix[i, :].sum() + self.confusion_matrix[:, i].sum() - intersection
            if union == 0:
                iou = np.nan
            else:
                iou = intersection / union
            ious.append(iou)
        ious = np.array(ious)
        mIoU = np.nanmean(ious)

        # Calcul de la précision, du rappel et du F1 par classe
        precision = np.zeros(self.num_classes)
        recall = np.zeros(self.num_classes)
        f1 = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            if (precision[i] + recall[i]) > 0:
                f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
            else:
                f1[i] = np.nan
        mean_precision = np.nanmean(precision)
        mean_recall = np.nanmean(recall)
        mean_f1 = np.nanmean(f1)

        # Stabilité temporelle : plus faible valeur = plus stable
        temporal_stability = (self.temporal_diff_sum / self.temporal_frames) if self.temporal_frames > 0 else np.nan

        return {
            "pixel_accuracy": pixel_accuracy,
            "ious": ious,
            "mIoU": mIoU,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
            "mean_f1": mean_f1,
            "temporal_stability": temporal_stability
        }
