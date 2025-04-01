import os
import sys
sys.path.append(os.path.abspath("src"))

import copy
import itertools
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from torch.multiprocessing import Manager
from tqdm import tqdm

from dataloader_sat_image import TemporalSatDataset, default_augmentation_fn, validation_fn
from loss_sat_images import temporal_semi_supervised_loss
from model_deeplabv3_plus import TemporalPlanetDeepLab
from metrics import MetricsEvaluator

# Variables globales
CSV_PATH        = "dataframe/df_merged_expanded.csv"
ALIGNMENT_CSV   = "dataframe/keyframes_alignment_geotorch.csv"
SEQ_LENGTH      = 12

def evaluate(model, loader, evaluator, device):
    model.eval()
    with torch.no_grad(), autocast(device_type='cuda'):
        for batch in tqdm(loader, desc="Evaluation", unit="batch"):
            X = batch["X"].to(device)
            Y = batch["Y"].to(device)
            preds = model(X)
            # Vérification et réarrangement des dimensions si nécessaire
            if preds.shape[0] != batch["mask_superv"].shape[1]:
                preds = preds.permute(1, 0, 2, 3, 4)
            T, B, C, H, W = preds.shape
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
    print(f"[Eval epoch {epoch}] PixelAcc: {pixel_acc:.4f} | mIoU: {miou:.4f} | Precision: {mean_precision:.4f} | Recall: {mean_recall:.4f} | F1: {mean_f1:.4f} | Ignored Px: {ignored_pixel}")

def train_fold(train_csv, val_csv,
               device, local_rank,
               num_epochs, batch_size, num_workers,
               lr, num_classes,
               lambda_temp=0.1,
               lambda_tversky=1.0,
               lambda_lovasz=1.0,
               alpha_tversky=0.7,
               beta_tversky=0.3,
               unfreeze_convlstm_epoch=5,
               unfreeze_encoder_epoch=10):
    """
    Entraînement d'un pli (fold) avec suivi multi‑objectif.
    Retourne (best_val_loss, best_val_metrics) où best_val_metrics contient,
    par exemple, la meilleure mIoU obtenue sur la validation.
    """
    train_dataset = TemporalSatDataset(train_csv, ALIGNMENT_CSV, default_augmentation_fn,
                                         seq_length=SEQ_LENGTH, random_subseq=True, split='train')
    val_dataset   = TemporalSatDataset(val_csv, ALIGNMENT_CSV, validation_fn,
                                         seq_length=SEQ_LENGTH * 4, random_subseq=False, split='val')
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                              sampler=val_sampler, num_workers=num_workers, pin_memory=True)
    
    # Initialisation du modèle avec gel progressif de certaines parties
    model = TemporalPlanetDeepLab(num_classes=num_classes).to(device)
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.convlstm.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scaler = GradScaler()
    
    # Exemple de poids de classes (à adapter)
    class_weights = torch.tensor([1.13, 0.78, 0.18, 11.43, 0.29, 1.0, 8.0], dtype=torch.float32)
    
    best_val_loss = float('inf')
    best_val_miou = 0.0
    best_val_metrics = None
    
    for epoch in range(1, num_epochs + 1):
        # Dégel progressif des modules
        if epoch == unfreeze_convlstm_epoch:
            for p in model.module.convlstm.parameters():
                p.requires_grad = True
            optimizer.add_param_group({"params": model.module.convlstm.parameters(), "lr": lr})
        
        if epoch == unfreeze_encoder_epoch:
            for p in model.module.encoder.parameters():
                p.requires_grad = True
            optimizer.add_param_group({"params": model.module.encoder.parameters(), "lr": lr * 0.1})
        
        model.train()
        train_sampler.set_epoch(epoch)
        
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{num_epochs}"):
            X = batch['X'].to(device)
            Y = batch['Y'].to(device)
            mask = batch['mask_superv'].to(device)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                preds = model(X)
                if preds.shape[0] != mask.shape[1]:
                    preds = preds.permute(1, 0, 2, 3, 4)
                loss = temporal_semi_supervised_loss(
                    preds,
                    {"Y": Y, "mask_superv": mask},
                    lambda_temp=lambda_temp,
                    lambda_tversky=lambda_tversky,
                    lambda_lovasz=lambda_lovasz,
                    num_classes=num_classes,
                    alpha_tversky=alpha_tversky,
                    beta_tversky=beta_tversky,
                    class_weights=class_weights
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        
        if local_rank == 0:
            print(f"[Epoch {epoch}] avg train loss: {avg_loss:.4f}")
            if epoch % 5 == 0 or epoch == num_epochs:
                evaluator = MetricsEvaluator(num_classes, ignore_index=255)
                metrics_val = evaluate(model.module, val_loader, evaluator, device)
                display_metrics(metrics_val, epoch)
                miou_val = metrics_val.get("mIoU", 0.0)
                
                # Suivi multi‑objectif :
                # - On sauvegarde le modèle si la loss diminue
                # - On met à jour la meilleure mIoU même si ce n'est pas accompagné d'une loss plus faible
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    best_val_metrics = metrics_val
                    torch.save(model.state_dict(), "best_model_fold_0.pth")
                if miou_val > best_val_miou:
                    best_val_miou = miou_val
    
    return best_val_loss, best_val_metrics

def ddp_worker(param_grid, results_list, device_params, train_csv, val_csv):
    """
    Parcourt la grille d'hyperparamètres et ajoute les résultats dans results_list.
    """
    rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{rank}")
    
    for combo in param_grid:
        if rank == 0:
            print("\n======================================")
            print(f"[DDP Rank 0] Testing config : {combo}")
        
        best_val_loss, best_val_metrics = train_fold(
            train_csv=train_csv,
            val_csv=val_csv,
            device=device,
            local_rank=rank,
            num_epochs=device_params["num_epochs"],
            batch_size=device_params["batch_size"],
            num_workers=device_params["num_workers"],
            lr=combo["lr"],
            num_classes=device_params["num_classes"],
            lambda_temp=combo["lambda_temp"],
            lambda_tversky=combo["lambda_tversky"],
            lambda_lovasz=combo["lambda_lovasz"],
            alpha_tversky=combo["alpha_tversky"],
            beta_tversky=combo["beta_tversky"],
            unfreeze_convlstm_epoch=device_params["unfreeze_convlstm_epoch"],
            unfreeze_encoder_epoch=device_params["unfreeze_encoder_epoch"]
        )
        dist.barrier()
        
        if rank == 0:
            best_val_miou = 0.0
            if best_val_metrics is not None:
                best_val_miou = best_val_metrics.get("mIoU", 0.0)
            res = {
                "config": combo,
                "best_val_loss": best_val_loss,
                "best_val_mIoU": best_val_miou,
            }
            results_list.append(res)
            print(f"[DDP Rank 0] => Config {combo} => best_val_loss={best_val_loss:.4f}, best_val_mIoU={best_val_miou:.4f}")

def main():
    # Initialisation du groupe DDP
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    # Création d'un split train/val par le processus de rang 0
    if local_rank == 0:
        df_full = pd.read_csv(CSV_PATH)
        df_train = df_full.sample(frac=0.8, random_state=42)
        df_val   = df_full.drop(df_train.index)
        os.makedirs("dataframe", exist_ok=True)
        df_train.to_csv("dataframe/train_fold_0.csv", index=False)
        df_val.to_csv("dataframe/val_fold_0.csv",   index=False)
    dist.barrier()
    
    train_csv = "dataframe/train_fold_0.csv"
    val_csv   = "dataframe/val_fold_0.csv"
    
    # Paramètres fixes pour l'entraînement
    device_params = {
        "num_epochs": 10,
        "batch_size": 4,
        "num_workers": 4,
        "num_classes": 7,
        "unfreeze_convlstm_epoch": 3,
        "unfreeze_encoder_epoch": 5
    }
    
    # Grille d'hyperparamètres à tester
    param_values = {
        "lr":             np.linspace(5e-5, 1e-4, num=2).tolist(),
        "lambda_temp":    np.arange(0.1, 1.2, 0.5).tolist(),
        "lambda_tversky": np.arange(0.1, 1.2, 0.5).tolist(),
        "lambda_lovasz":  np.arange(0.1, 1.2, 0.5).tolist(),
        "alpha_tversky":  np.round(np.arange(0.1, 1.2, 0.5), 2).tolist(),
        "beta_tversky":   np.round(np.arange(0.1, 1.2, 0.5), 2).tolist()
    }
    
    all_keys = list(param_values.keys())
    combos = list(itertools.product(*(param_values[k] for k in all_keys)))
    param_grid = []
    for c in combos:
        d = {}
        for k, v in zip(all_keys, c):
            d[k] = v
        param_grid.append(d)
    
    if local_rank == 0:
        print(f"Nombre total de configurations à tester : {len(param_grid)}")
    
    manager = Manager()
    results_list = manager.list()
    
    # Exécution du grid search via le ddp_worker
    ddp_worker(param_grid, results_list, device_params, train_csv, val_csv)
    
    if local_rank == 0:
        final_res = list(results_list)
        df_res = []
        for item in final_res:
            row = copy.deepcopy(item["config"])
            row["best_val_loss"] = item["best_val_loss"]
            row["best_val_mIoU"] = item["best_val_mIoU"]
            df_res.append(row)
        df_res = pd.DataFrame(df_res)
        
        os.makedirs("grid_search_out", exist_ok=True)
        out_csv = "grid_search_out/grid_search_results.csv"
        df_res.to_csv(out_csv, index=False)
        
        print("\n===== GRID SEARCH FINISHED =====")
        print("All results:\n", df_res)
        
        # Post-filtrage multi‑objectif : on impose un seuil minimal sur la mIoU
        min_miou = 0.40  # seuil minimal (à adapter)
        df_filtered = df_res[df_res["best_val_mIoU"] >= min_miou]
        if len(df_filtered) == 0:
            print(f"Aucune config n'atteint mIoU >= {min_miou}. On choisit la config avec la mIoU maximale.")
            best_idx = df_res["best_val_mIoU"].idxmax()
            best_row = df_res.loc[best_idx]
        else:
            best_idx = df_filtered["best_val_loss"].idxmin()
            best_row = df_filtered.loc[best_idx]
        
        print("BEST CONFIG:", dict(best_row.drop(["best_val_loss", "best_val_mIoU"])))
        print("BEST VAL LOSS:", best_row["best_val_loss"])
        print("BEST VAL mIoU:", best_row["best_val_mIoU"])
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
