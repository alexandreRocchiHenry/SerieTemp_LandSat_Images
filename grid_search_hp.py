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

from dataloader_sat_image import TemporalSatDataset, default_augmentation_fn, validation_fn
from loss_sat_images import temporal_semi_supervised_loss
from model_deeplabv3_plus import TemporalPlanetDeepLab
from metrics import MetricsEvaluator

# ------------------------------------------------------------------------
# Paramètres globaux
# ------------------------------------------------------------------------
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
    pixel_acc     = metrics.get("pixel_accuracy", 0)
    miou          = metrics.get("mIoU", 0)
    mean_precision = metrics.get("mean_precision", 0)
    mean_recall   = metrics.get("mean_recall", 0)
    mean_f1       = metrics.get("mean_f1", 0)
    ignored_pixel = metrics.get("ignored_pixel_count", 0)
    print(f"[Eval epoch {epoch}] PixelAcc: {pixel_acc:.4f} | mIoU: {miou:.4f} "
          f"| Precision: {mean_precision:.4f} | Recall: {mean_recall:.4f} "
          f"| F1: {mean_f1:.4f} | Ignored Px: {ignored_pixel}")

class FreezeUnfreezeCallback(Callback):
    """Callback pour geler/dégeler l’encoder et convlstm à des epochs précises."""
    def __init__(self, unfreeze_convlstm_epoch=3, unfreeze_encoder_epoch=5):
        super().__init__()
        self.unfreeze_convlstm_epoch = unfreeze_convlstm_epoch
        self.unfreeze_encoder_epoch = unfreeze_encoder_epoch
    
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
        # Par défaut (entièrement gelé, puis dégel progressif)
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        for p in self.model.convlstm.parameters():
            p.requires_grad = False
        for p in self.model.classifier.parameters():
            p.requires_grad = True
        
        self.criterion = temporal_semi_supervised_loss
        self.evaluator = MetricsEvaluator(num_classes=self.hparams.num_classes,
                                          ignore_index=255)
        
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

def train_single_config(
    train_dataset,
    val_dataset,
    device_params,
    combo
):
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
        patience=5,
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

def main():
    seed_everything(42)
    
    # Prepare Data
    df_full = pd.read_csv(CSV_PATH)
    df_train = df_full.sample(frac=0.8)
    df_val   = df_full.drop(df_train.index)
    
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
    
    device_params = {
        "num_epochs": 20,
        "batch_size": 4,
        "num_workers": 4,
        "num_classes": 7,
        "unfreeze_convlstm_epoch": 1,
        "unfreeze_encoder_epoch": 2,
    }
    
    # ------------------------------------------------------------------
    # STAGE 1 : grille autour des meilleures valeurs observées apres deja une recherche exhaustive plus lagre
    # ------------------------------------------------------------------
    param_values_coarse = {
        "lr":             [5e-5],
        "lambda_temp":    [0.05, 0.1, 0.15],
        "lambda_tversky": [0.8, 1.0, 1.2],
        "lambda_lovasz":  [0.8, 1.0, 1.2],
        "alpha_tversky":  [0.65, 0.7, 0.75],
        "beta_tversky":   [0.25, 0.35],
    }
    all_keys_coarse = list(param_values_coarse.keys())
    combos_coarse = list(itertools.product(*(param_values_coarse[k] for k in all_keys_coarse)))
    
    param_grid_coarse = []
    for c in combos_coarse:
        d = {}
        for k, v in zip(all_keys_coarse, c):
            d[k] = v
        param_grid_coarse.append(d)
    
    print(f"===== STAGE 1 : COARSE GRID (nb={len(param_grid_coarse)}) =====")
    results_list_coarse = []
    
    for combo in param_grid_coarse:
        print("\n======================================")
        print(f"Testing config : {combo}")
        model, val_miou = train_single_config(train_dataset, val_dataset, device_params, combo)
        print(f"=> Config {combo} => best_val_mIoU={val_miou:.4f}")
        results_list_coarse.append({
            "config": combo,
            "best_val_mIoU": val_miou,
        })
    
    df_coarse = []
    for item in results_list_coarse:
        row = copy.deepcopy(item["config"])
        row["best_val_mIoU"] = item["best_val_mIoU"]
        df_coarse.append(row)
    df_coarse = pd.DataFrame(df_coarse)
    df_coarse.to_csv("grid_search_out_coarse.csv", index=False)
    
    print("\n===== COARSE SEARCH FINISHED =====")
    print("All results:\n", df_coarse)
    best_idx = df_coarse["best_val_mIoU"].idxmax()
    best_row = df_coarse.loc[best_idx]
    print("\nBest config (coarse) = ", dict(best_row))

    best_lr   = best_row["lr"]
    best_temp = best_row["lambda_temp"]
    best_tv   = best_row["lambda_tversky"]
    best_lov  = best_row["lambda_lovasz"]
    best_at   = best_row["alpha_tversky"]
    best_bt   = best_row["beta_tversky"]
    
    def around(v, step=0.2, n=3):
        start = v - step
        end   = v + step
        return np.linspace(start, end, n).tolist()
    
    param_values_fine = {
        "lr":             [best_lr * 0.8, best_lr, best_lr * 1.2],
        "lambda_temp":    around(best_temp, 0.4, 3),
        "lambda_tversky": around(best_tv,   0.4, 3),
        "lambda_lovasz":  around(best_lov,  0.4, 3),
        "alpha_tversky":  around(best_at,   0.1, 3),
        "beta_tversky":   around(best_bt,   0.1, 3),
    }
    np.save("param_values_fine.npy", param_values_fine, allow_pickle=True)
    
    print(f"\nOn a généré param_values_fine (stocké dans param_values_fine.npy).")
    
    # ------------------------------------------------------------------
    # STAGE 2 : on affine la recherche
    # ------------------------------------------------------------------
    if os.path.exists("param_values_fine.npy"):
        param_values_fine_loaded = np.load("param_values_fine.npy", allow_pickle=True).item()
        all_keys_fine = list(param_values_fine_loaded.keys())
        combos_fine = list(itertools.product(*(param_values_fine_loaded[k] for k in all_keys_fine)))
        param_grid_fine = []
        for c in combos_fine:
            d = {}
            for k, v in zip(all_keys_fine, c):
                d[k] = float(v)
            param_grid_fine.append(d)
        
        print(f"\n===== STAGE 2 : FINE GRID (nb={len(param_grid_fine)}) =====")
        results_list_fine = []
        
        for combo in param_grid_fine:
            print("\n======================================")
            print(f"Testing config : {combo}")
            model, val_miou, best_ckpt_path = train_single_config(train_dataset, val_dataset, device_params, combo)
            print(f"=> Config {combo} => best_val_mIoU={val_miou:.4f}")
            results_list_fine.append({
                "config": combo,
                "best_val_mIoU": val_miou,
                "best_ckpt_path": best_ckpt_path
            })
        
        df_fine = []
        for item in results_list_fine:
            row = copy.deepcopy(item["config"])
            row["best_val_mIoU"] = item["best_val_mIoU"]
            row["best_ckpt_path"] = item["best_ckpt_path"]
            df_fine.append(row)
        df_fine = pd.DataFrame(df_fine)
        df_fine.to_csv("grid_search_out_fine.csv", index=False)
        
        print("\n===== FINE SEARCH FINISHED =====")
        print("All fine results:\n", df_fine)
        best_idx_fine = df_fine["best_val_mIoU"].idxmax()
        best_row_fine = df_fine.loc[best_idx_fine]
        print("Best config (fine) => ", dict(best_row_fine))
    else:
        print("Pas de param_values_fine.npy, STAGE 2 ignoré.")
    
if __name__ == "__main__":
    main()