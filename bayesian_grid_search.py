
import os
import sys
sys.path.append(os.path.abspath("src"))
import copy
import itertools
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.multiprocessing import Manager
from torch.utils.data import DataLoader, DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy

# optimisation bayésienne
import optuna

from dataloader_sat_image import TemporalSatDataset, default_augmentation_fn, validation_fn
from loss_sat_images import temporal_semi_supervised_loss
from model_deeplabv3_plus import TemporalPlanetDeepLab
from metrics import MetricsEvaluator

CSV_PATH        = "dataframe/df_merged_expanded.csv"
ALIGNMENT_CSV   = "dataframe/keyframes_alignment_geotorch.csv"
SEQ_LENGTH      = 12

def evaluate_segmentation(model, loader, evaluator, device):
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
    pixel_acc      = metrics.get("pixel_accuracy", 0)
    miou           = metrics.get("mIoU", 0)
    mean_precision = metrics.get("mean_precision", 0)
    mean_recall    = metrics.get("mean_recall", 0)
    mean_f1        = metrics.get("mean_f1", 0)
    ignored_pixel  = metrics.get("ignored_pixel_count", 0)
    print(f"[Eval epoch {epoch}] PixelAcc: {pixel_acc:.4f} | mIoU: {miou:.4f} "
          f"| Precision: {mean_precision:.4f} | Recall: {mean_recall:.4f} "
          f"| F1: {mean_f1:.4f} | Ignored Px: {ignored_pixel}")

class FreezeUnfreezeCallback(Callback):
    def __init__(self, unfreeze_convlstm_epoch=3, unfreeze_encoder_epoch=5):
        super().__init__()
        self.unfreeze_convlstm_epoch = unfreeze_convlstm_epoch
        self.unfreeze_encoder_epoch  = unfreeze_encoder_epoch
    
    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch == self.unfreeze_convlstm_epoch:
            for p in pl_module.model.convlstm.parameters():
                p.requires_grad = True
            print(f"[Epoch {epoch}] Dégel convlstm.")
        
        if epoch == self.unfreeze_encoder_epoch:
            for p in pl_module.model.encoder.parameters():
                p.requires_grad = True
            for g in pl_module.optimizers().param_groups:
                if any(id(p) in [id(pp) for pp in pl_module.model.encoder.parameters()] for p in g["params"]):
                    g["lr"] = g["lr"] * 0.1
            print(f"[Epoch {epoch}] Dégel encoder (LR réduit).")

class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch=10, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_epoch_end(trainer, pl_module)

class LitTemporalSeg(pl.LightningModule):
    def __init__(
        self, 
        num_classes=7,
        lr=1e-4,
        lambda_temp=0.1,
        lambda_tversky=1.0,
        lambda_lovasz=1.0,
        alpha_tversky=0.7,
        beta_tversky=0.3
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = TemporalPlanetDeepLab(num_classes=self.hparams.num_classes)
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        for p in self.model.convlstm.parameters():
            p.requires_grad = False
        for p in self.model.classifier.parameters():
            p.requires_grad = True
        
        self.criterion = temporal_semi_supervised_loss
        self.evaluator = MetricsEvaluator(num_classes=self.hparams.num_classes, ignore_index=255)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        X = batch['X']
        Y = batch['Y']
        mask = batch['mask_superv']
        preds = self(X)
        if preds.shape[0] != mask.shape[1]:
            preds = preds.permute(1, 0, 2, 3, 4)
        loss = self.criterion(
            preds,
            {"Y": Y, "mask_superv": mask},
            lambda_temp=self.hparams.lambda_temp,
            lambda_tversky=self.hparams.lambda_tversky,
            lambda_lovasz=self.hparams.lambda_lovasz,
            num_classes=self.hparams.num_classes,
            alpha_tversky=self.hparams.alpha_tversky,
            beta_tversky=self.hparams.beta_tversky
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=X.size(0))
        return loss
    
    def validation_step(self, batch, batch_idx):
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
        metrics = self.evaluator.compute()
        self.evaluator.reset()
        self.log("val_mIoU", metrics.get("mIoU", 0.0), prog_bar=True)
        self.log("val_loss", 0.0)
        display_metrics(metrics, self.current_epoch + 1)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr
        )
        return optimizer

def train_single_config(train_dataset, val_dataset, device_params, combo):
    """Entraîne et évalue un modèle Lightning pour une config hyperparam."""
    model = LitTemporalSeg(
        num_classes=device_params["num_classes"],
        lr=combo["lr"],
        lambda_temp=combo["lambda_temp"],
        lambda_tversky=combo["lambda_tversky"],
        lambda_lovasz=combo["lambda_lovasz"],
        alpha_tversky=combo["alpha_tversky"],
        beta_tversky=combo["beta_tversky"]
    )
    
    callback_freeze = FreezeUnfreezeCallback(
        unfreeze_convlstm_epoch=device_params["unfreeze_convlstm_epoch"],
        unfreeze_encoder_epoch=device_params["unfreeze_encoder_epoch"]
    )
    early_stop_callback = DelayedEarlyStopping(
        start_epoch=10,
        monitor="val_mIoU",
        mode="max",
        patience=11,
        verbose=True
    )
    
    trainer = Trainer(
        max_epochs=device_params["num_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1 if torch.cuda.is_available() else None,
        strategy=DDPStrategy(find_unused_parameters=True),
        accumulate_grad_batches=12,
        callbacks=[callback_freeze, early_stop_callback],
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
    best_val_miou = trainer.callback_metrics.get("val_mIoU", 0.0).item() if trainer.callback_metrics else 0.0
    return model, best_val_miou

def objective(trial, train_dataset, val_dataset, device_params):
    lr             = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    lambda_temp    = trial.suggest_float("lambda_temp", 0.01, 0.2, step=0.01)
    lambda_tversky = trial.suggest_float("lambda_tversky", 0.6, 1.4, step=0.1)
    lambda_lovasz  = trial.suggest_float("lambda_lovasz", 0.6, 1.4, step=0.1)
    alpha_tversky  = trial.suggest_float("alpha_tversky", 0.5, 0.8, step=0.05)
    beta_tversky   = trial.suggest_float("beta_tversky", 0.2, 0.4, step=0.05)

    combo = {
        "lr": lr,
        "lambda_temp": lambda_temp,
        "lambda_tversky": lambda_tversky,
        "lambda_lovasz": lambda_lovasz,
        "alpha_tversky": alpha_tversky,
        "beta_tversky": beta_tversky,
    }
    _, val_miou = train_single_config(train_dataset, val_dataset, device_params, combo)
    return val_miou

def main():
    seed_everything(42)
    
    df_full = pd.read_csv(CSV_PATH)
    df_train = df_full.sample(frac=0.8)
    df_val = df_full.drop(df_train.index)
    
    train_csv = "dataframe/train_fold_0.csv"
    val_csv   = "dataframe/val_fold_0.csv"
    os.makedirs("dataframe", exist_ok=True)
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv,   index=False)
    
    train_dataset = TemporalSatDataset(
        train_csv, ALIGNMENT_CSV, default_augmentation_fn,
        seq_length=SEQ_LENGTH, random_subseq=True, split='train'
    )
    val_dataset = TemporalSatDataset(
        val_csv, ALIGNMENT_CSV, validation_fn,
        seq_length=SEQ_LENGTH * 4, random_subseq=False, split='val'
    )
    
    # paramètres de base
    device_params = {
        "num_epochs": 30,
        "batch_size": 4,
        "num_workers": 4,
        "num_classes": 7,
        "unfreeze_convlstm_epoch": 5,
        "unfreeze_encoder_epoch": 10,
    }
    # paramètres de recherche
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, device_params), n_trials=10)
    
    print("\n===== BAYESIAN SEARCH FINISHED =====")
    print("Best trial:")
    best_trial = study.best_trial
    print("Value (val_mIoU):", best_trial.value)
    print("Params:", best_trial.params)

if __name__ == "__main__":
    main()
