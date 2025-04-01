import numpy as np
import torch
import torch.nn.functional as F
class MetricsEvaluator:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.total_correct = 0
        self.total_valid = 0
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        self.gt_pixel_counts = np.zeros(self.num_classes, dtype=np.int64)

        self.ignored_pixel_count = 0
        self.temporal_diff_sum = 0.0
        self.temporal_frames = 0

    def update(self, pred, target):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        if pred.ndim == 3:
            B = pred.shape[0]
            for b in range(B):
                self._update_frame(pred[b], target[b])
        else:
            self._update_frame(pred, target)

    def _update_frame(self, pred, target):
        valid_mask = (target != self.ignore_index)
        ignored_mask = (target == self.ignore_index)
        num_ignored = ignored_mask.sum()
        
        if valid_mask.sum() == 0:
            return
        else:
            self.ignored_pixel_count += num_ignored
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        self.total_correct += (pred_valid == target_valid).sum()
        self.total_valid += valid_mask.sum()
        
      
        for cls in range(self.num_classes):
            self.gt_pixel_counts[cls] += (target_valid == cls).sum()
        
        inds = self.num_classes * target_valid + pred_valid
        cm_update = np.bincount(inds, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += cm_update

    def update_temporal(self, pred_seq):
        if isinstance(pred_seq, torch.Tensor):
            pred_seq = pred_seq.cpu().numpy()
        T = pred_seq.shape[0]
        if T < 2:
            return
        diff = 0.0
        count = 0
        for t in range(T - 1):
            frame_diff = np.mean(np.abs(pred_seq[t+1] - pred_seq[t]))
            diff += frame_diff
            count += 1
        self.temporal_diff_sum += diff
        self.temporal_frames += count

    def compute(self):
        pixel_accuracy = self.total_correct / self.total_valid if self.total_valid > 0 else 0.0

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
            "temporal_stability": temporal_stability,
            "gt_pixel_counts": self.gt_pixel_counts,
            "ignored_pixel_count": self.ignored_pixel_count
        }
