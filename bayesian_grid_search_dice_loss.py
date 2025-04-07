import os
import sys
sys.path.append(os.path.abspath("src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy

import optuna

# ------------------------------------------------------------------
# Imports locaux (à adapter selon votre organisation de fichiers)
# ------------------------------------------------------------------
from dataloader_sat_image import TemporalSatDataset, default_augmentation_fn, validation_fn
from model_deeplabv3_plus import TemporalPlanetDeepLab
from metrics import MetricsEvaluator

# ------------------------------------------------------------------
# Paramètres de vos CSV
# ------------------------------------------------------------------
CSV_PATH        = "dataframe/df_merged_expanded.csv"
ALIGNMENT_CSV   = "dataframe/keyframes_alignment_geotorch.csv"
SEQ_LENGTH      = 12


# ------------------------------------------------------------------
# FONCTIONS UTILES
# ------------------------------------------------------------------
def evaluate_segmentation(model, loader, evaluator, device):
    """Évalue rapidement la segmentation sur un DataLoader et retourne les métriques."""
    model.eval()
    for batch in loader:
        X = batch["X"].to(device)
        Y = batch["Y"].to(device)
        preds = model(X)
        if preds.shape[0] != batch["mask_superv"].shape[1]:
            preds = preds.permute(1, 0, 2, 3, 4)
        T, B, _, _, _ = preds.shape
        for t in range(T):
            for b in range(B):
                pred_mask = torch.argmax(preds[t, b], dim=0)
                evaluator.update(pred_mask, Y[b, t])
    return evaluator.compute()

def display_metrics(metrics, epoch):
    """Affiche les métriques formatées."""
    pixel_acc      = metrics.get("pixel_accuracy", 0)
    miou           = metrics.get("mIoU", 0)
    mean_precision = metrics.get("mean_precision", 0)
    mean_recall    = metrics.get("mean_recall", 0)
    mean_f1        = metrics.get("mean_f1", 0)
    ignored_pixel  = metrics.get("ignored_pixel_count", 0)
    print(
        f"[Eval epoch {epoch}] "
        f"PixelAcc: {pixel_acc:.4f} | mIoU: {miou:.4f} | Precision: {mean_precision:.4f} "
        f"| Recall: {mean_recall:.4f} | F1: {mean_f1:.4f} | Ignored Px: {ignored_pixel}"
    )


class FreezeUnfreezeCallback(Callback):
    """
    Exemple de callback qui gèle certains modules au début, 
    puis les déverrouille (“unfreeze”) à des époques précises.
    """
    def __init__(self, unfreeze_convlstm_epoch=3, unfreeze_encoder_epoch=5):
        super().__init__()
        self.unfreeze_convlstm_epoch = unfreeze_convlstm_epoch
        self.unfreeze_encoder_epoch  = unfreeze_encoder_epoch
    
    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1

        # Dégeler le convlstm
        if epoch == self.unfreeze_convlstm_epoch:
            for p in pl_module.model.convlstm.parameters():
                p.requires_grad = True
            print(f"[Epoch {epoch}] Dégel convlstm.")
        
        # Dégeler l'encoder + baisser le LR pour ces params
        if epoch == self.unfreeze_encoder_epoch:
            for p in pl_module.model.encoder.parameters():
                p.requires_grad = True

            # Ajuste le learning rate uniquement pour l'encoder
            for g in pl_module.optimizers().param_groups:
                # On vérifie si un paramètre du groupe est dans l'encoder
                if any(id(p) in [id(pp) for pp in pl_module.model.encoder.parameters()] for p in g["params"]):
                    g["lr"] = g["lr"] * 0.1
            print(f"[Epoch {epoch}] Dégel encoder (LR réduit).")

class DelayedEarlyStopping(EarlyStopping):
    """
    Callback EarlyStopping qui n’est actif qu’après un certain nombre d’époques.
    """
    def __init__(self, start_epoch=10, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_epoch_end(trainer, pl_module)


# ------------------------------------------------------------------
# PERTE DICE + CONSISTANCE TEMPORELLE
# ------------------------------------------------------------------
class WeightedDiceLoss(nn.Module):
    """
    Implémentation d’un Dice Loss (1 - Dice) pour gérer un déséquilibre de classes,
    avec support d’un ignore_index et de poids de classe.
    """
    def __init__(self, num_classes, ignore_index=255, class_weights=None, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_weights = class_weights
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] (sorties du modèle, non-normalisées)
        targets: [B, H, W]  (index de classe par pixel)
        """
        # Convertit logits en probabilités
        probs = F.softmax(logits, dim=1)

        # Crée un masque pour ignorer l'ignore_index
        mask = (targets != self.ignore_index)
        if mask.sum() == 0:
            return torch.tensor(0., device=logits.device)
        
        # On met la valeur 0 là où c'est ignoré, pour éviter d'introduire de la classe invalid
        targets_clamped = targets.clone()
        targets_clamped[~mask] = 0
        
        # One-hot des cibles
        one_hot = F.one_hot(targets_clamped, self.num_classes).permute(0, 3, 1, 2).float()

        # On applique le masque sur la prédiction et la cible
        one_hot = one_hot * mask.unsqueeze(1)
        probs   = probs   * mask.unsqueeze(1)

        # Calcul par classe
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        union        = probs.sum(dims) + one_hot.sum(dims)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        # Poids de classe
        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)[:self.num_classes]
            dice_loss = dice_loss * w
        
        return dice_loss.mean()


def temporal_semi_supervised_dice_loss(
    preds: torch.Tensor,
    sample: dict,
    dice_criterion,
    lambda_temp: float = 1.0
) -> torch.Tensor:
    """
    Utilise le Dice Loss pour la partie supervisée + une consistance temporelle (KL symétrique) 
    entre frames t et t+1 pour la partie non-supervisée.
    Args:
        preds:  [T, B, C, H, W] (sorties du réseau sur la séquence)
        sample: {"Y": [B, T, H, W], "mask_superv": [B, T]} 
                -> Y: masques de segmentation
                -> mask_superv: booleen indiquant si la frame t est supervisée
        dice_criterion: instance de WeightedDiceLoss
        lambda_temp: coefficient pour la consistance temporelle
    """
    # On s’attend à [T, B, ...], donc si c’est [B, T, ...] on permute
    if preds.shape[0] != sample["mask_superv"].shape[1]:
        preds = preds.permute(1, 0, 2, 3, 4)

    T, B, C, H, W = preds.shape
    Y = sample["Y"].permute(1, 0, 2, 3)              # [T, B, H, W]
    mask = sample["mask_superv"].permute(1, 0).bool()  # [T, B]

    # ----- Perte supervisée (Dice) -----
    sup_loss = torch.tensor(0., device=preds.device)
    valid_count = 0

    for t in range(T):
        valid = mask[t]         # [B]
        if valid.any():
            logits_t  = preds[t, valid]   # [n_valid, C, H, W]
            targets_t = Y[t, valid]       # [n_valid, H, W]
            loss_dice = dice_criterion(logits_t, targets_t)
            sup_loss += loss_dice
            valid_count += valid.sum()

    sup_loss = sup_loss / max(valid_count, 1)

    # ----- Consistance temporelle (KL) -----
    preds_soft = F.softmax(preds, dim=2)  # [T, B, C, H, W]
    temp_loss = torch.tensor(0., device=preds.device)
    if T > 1:
        for t in range(T - 1):
            p_t   = preds_soft[t]   # [B, C, H, W]
            p_t1  = preds_soft[t+1] # [B, C, H, W]
            log_p = torch.log(p_t  + 1e-8)
            log_q = torch.log(p_t1 + 1e-8)
            kl_pq = F.kl_div(log_p, p_t1, reduction='batchmean')
            kl_qp = F.kl_div(log_q, p_t,  reduction='batchmean')
            temp_loss += (kl_pq + kl_qp) / 2
        temp_loss /= (T - 1)

    return sup_loss + lambda_temp * temp_loss


# ------------------------------------------------------------------
# MODULE LIGHTNING
# ------------------------------------------------------------------
class LitTemporalSeg(pl.LightningModule):
    """
    Module Lightning qui entraîne un réseau de segmentation temporelle 
    (ex: DeepLab + ConvLSTM) avec WeightedDiceLoss et consistance temporelle.
    """
    def __init__(
        self, 
        num_classes=7,
        lr=1e-4,
        lambda_temp=0.1
    ):
        """
        Args:
            num_classes : nombre de classes
            lr : learning rate initial
            lambda_temp : pondération de la consistance temporelle
        """
        super().__init__()
        self.save_hyperparameters()

        # Modèle de segmentation temporelle
        self.model = TemporalPlanetDeepLab(num_classes=self.hparams.num_classes)

        # On fige l’encoder et le convlstm pour commencer (ils seront dégelés plus tard)
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        for p in self.model.convlstm.parameters():
            p.requires_grad = False
        for p in self.model.classifier.parameters():
            p.requires_grad = True
        
        # Loss Dice

        class_weights = torch.tensor([1.13, 0.78, 0.18, 11.43, 0.29, 1.0, 8.0], dtype=torch.float32)
        self.dice_loss_fn = WeightedDiceLoss(self.hparams.num_classes, class_weights=class_weights)


        # Métriques
        self.evaluator = MetricsEvaluator(num_classes=self.hparams.num_classes, ignore_index=255)
    
    def forward(self, x):
        """Forward standard du modèle."""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        Entraînement: on calcule la perte (Dice + consistance temporelle) sur les frames supervisées.
        """
        X = batch['X']
        Y = batch['Y']
        mask = batch['mask_superv']  # booleen indiquant si la frame est supervisée ou non
        preds = self(X)

        # S’assure que [T, B] si besoin
        if preds.shape[0] != mask.shape[1]:
            preds = preds.permute(1, 0, 2, 3, 4)

        # Calcule la perte
        loss = temporal_semi_supervised_dice_loss(
            preds,
            {"Y": Y, "mask_superv": mask},
            dice_criterion=self.dice_loss_fn,
            lambda_temp=self.hparams.lambda_temp
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=X.size(0))
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation: on utilise seulement les prédictions finales pour mettre à jour l’évaluateur.
        """
        X = batch['X']
        Y = batch['Y']
        preds = self(X)

        if preds.shape[0] != batch["mask_superv"].shape[1]:
            preds = preds.permute(1, 0, 2, 3, 4)

        T, B, _, _, _ = preds.shape
        for t in range(T):
            for b in range(B):
                pred_mask = torch.argmax(preds[t, b], dim=0)
                self.evaluator.update(pred_mask, Y[b, t])
    
    def on_validation_epoch_end(self):
        """
        On récupère les métriques agrégées, on les log, et on réinitialise l’évaluateur.
        """
        metrics = self.evaluator.compute()
        self.evaluator.reset()
        self.log("val_mIoU", metrics.get("mIoU", 0.0), prog_bar=True)
        self.log("val_loss", 0.0)  # facultatif, parfois on ne calcule pas de vrai val_loss
        display_metrics(metrics, self.current_epoch + 1)
    
    def configure_optimizers(self):
        """
        On utilise typiquement AdamW, on retourne l’optimisateur (et scheduler éventuel).
        """
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr
        )
        return optimizer


# ------------------------------------------------------------------
# BOUCLE D’ENTRAÎNEMENT
# ------------------------------------------------------------------
def train_single_config(train_dataset, val_dataset, device_params, combo):
    """
    Entraîne et évalue un modèle Lightning pour une config hyperparam.
    combo est un dict avec par ex. {"lr": 1e-4, "lambda_temp": 0.1, ...}
    """
    model = LitTemporalSeg(
        num_classes=device_params["num_classes"],
        lr=combo["lr"],
        lambda_temp=combo["lambda_temp"]
    )
    
    # Callback pour geler/dégeler l’encoder et convlstm
    callback_freeze = FreezeUnfreezeCallback(
        unfreeze_convlstm_epoch=device_params["unfreeze_convlstm_epoch"],
        unfreeze_encoder_epoch=device_params["unfreeze_encoder_epoch"]
    )

    # Exemple (commenté) d’early stopping retardé
    # early_stop_callback = DelayedEarlyStopping(
    #     start_epoch=10,
    #     monitor="val_mIoU",
    #     mode="max",
    #     patience=11,
    #     verbose=True
    # )

    trainer = Trainer(
        max_epochs=device_params["num_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1 if torch.cuda.is_available() else None,
        strategy=DDPStrategy(find_unused_parameters=True),
        accumulate_grad_batches=5,
        callbacks=[callback_freeze],
        log_every_n_steps=10,
        precision=16,
    )
    
    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            train_dataset,
            batch_size=device_params["batch_size"],
            num_workers=device_params["num_workers"],
            shuffle=True
        ),
        val_dataloaders=DataLoader(
            val_dataset,
            batch_size=device_params["batch_size"],
            num_workers=device_params["num_workers"]
        ),
    )

    # Récupération de la valeur de mIoU en fin d’entraînement
    best_val_miou = trainer.callback_metrics.get("val_mIoU", 0.0).item() if trainer.callback_metrics else 0.0
    return model, best_val_miou


def objective(trial, train_dataset, val_dataset, device_params):
    """
    Fonction d’objectif Optuna pour tester différentes combinaisons (lr, lambda_temp, etc.).
    """
    lr          = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    lambda_temp = trial.suggest_float("lambda_temp", 0.01, 0.2, step=0.01)

    combo = {
        "lr": lr,
        "lambda_temp": lambda_temp,
    }
    _, val_miou = train_single_config(train_dataset, val_dataset, device_params, combo)
    return val_miou


def main():
    seed_everything(42)
    
    # Lecture du CSV
    df_full = pd.read_csv(CSV_PATH)
    df_train = df_full.sample(frac=0.8)
    df_val   = df_full.drop(df_train.index)
    
    # Sauvegarde en local (optionnel, si vous souhaitez splitter différemment)
    train_csv = "dataframe/train_fold_0.csv"
    val_csv   = "dataframe/val_fold_0.csv"
    os.makedirs("dataframe", exist_ok=True)
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv,   index=False)
    
    # Construction des datasets
    train_dataset = TemporalSatDataset(
        train_csv, 
        ALIGNMENT_CSV, 
        default_augmentation_fn,
        seq_length=SEQ_LENGTH, 
        random_subseq=True, 
        split='train'
    )
    val_dataset = TemporalSatDataset(
        val_csv, 
        ALIGNMENT_CSV, 
        validation_fn,
        seq_length=SEQ_LENGTH * 4, 
        random_subseq=False, 
        split='val'
    )
    
    # Paramètres globaux
    device_params = {
        "num_epochs": 30,
        "batch_size": 4,
        "num_workers": 4,
        "num_classes": 7,
        "unfreeze_convlstm_epoch": 5,
        "unfreeze_encoder_epoch": 10,
    }

    # Lancement d’une recherche bayésienne
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, device_params), n_trials=10)
    
    # Exemple d’enregistrement des résultats
    df_study = pd.DataFrame([
        {**s.params, "value": s.value} for s in study.trials
    ])
    df_study.to_csv("bayesian_search_results.csv", index=False)

    print("\n===== BAYESIAN SEARCH FINISHED =====")
    print("Best trial:")
    best_trial = study.best_trial
    print("Value (val_mIoU):", best_trial.value)
    print("Params:", best_trial.params)


if __name__ == "__main__":
    main()
