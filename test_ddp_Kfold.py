import os
import sys
sys.path.append(os.path.abspath("src"))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import GroupKFold

from dataloader_sat_image import TemporalSatDataset, default_augmentation_fn, validation_fn
from loss_sat_images import temporal_semi_supervised_loss
from model_deeplabv3_plus import TemporalPlanetDeepLab
from metrics import MetricsEvaluator

def evaluate(model, loader, evaluator, device):
    model.eval()
    # Utilisation de torch.amp pour accélérer l'évaluation
    with torch.no_grad(), autocast(device_type='cuda'):
        for batch in tqdm(loader, desc="Evaluation", unit="batch"):
            X = batch["X"].to(device)
            Y = batch["Y"].to(device)
            preds = model(X)
            # On vérifie que les dimensions sont conformes (selon l'ordre des axes)
            if preds.shape[0] != batch["mask_superv"].shape[1]:
                preds = preds.permute(1, 0, 2, 3, 4)
            T, B, C, H, W = preds.shape
            for t in range(T):
                for b in range(B):
                    pred_mask = torch.argmax(preds[t, b], dim=0)
                    evaluator.update(pred_mask, Y[b, t])
    return evaluator.compute()

def display_metrics(metrics, epoch, fold):
    pixel_acc     = metrics.get("pixel_accuracy", 0)
    miou          = metrics.get("mIoU", 0)
    mean_precision = metrics.get("mean_precision", 0)
    mean_recall   = metrics.get("mean_recall", 0)
    mean_f1       = metrics.get("mean_f1", 0)
    ignored_pixel = metrics.get("ignored_pixel_count", 0)
    print(f"Fold {fold} - Epoch {epoch} validation metrics: \nPixel Acc: {pixel_acc:.4f} | mIoU: {miou:.4f} | Mean Precision: {mean_precision:.4f} | Mean Recall: {mean_recall:.4f} | Mean F1: {mean_f1:.4f} | Ignored Pixels: {ignored_pixel}")
    


def train_fold(train_csv, val_csv, fold, device, local_rank, num_epochs, batch_size, num_workers, num_classes, lr):
    
    # Pour 7 classes issues du papier, et si vous avez num_classes=8, on ajoute une valeur par défaut pour la 8ᵉ classe.
    class_weights = torch.tensor([1.13, 0.78, 0.18, 11.43, 0.29, 1.0, 8.0], dtype=torch.float32)
    
    # Création des datasets à partir des CSV spécifiques au pli
    train_dataset = TemporalSatDataset(train_csv, ALIGNMENT_CSV, default_augmentation_fn,
                                         seq_length=SEQ_LENGTH, random_subseq=True, split='train')
    val_dataset = TemporalSatDataset(val_csv, ALIGNMENT_CSV, validation_fn,
                                       seq_length=SEQ_LENGTH * 4, random_subseq=False, split='val')
    
    # Création des DistributedSamplers pour répartir les données entre les processus
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=num_workers, pin_memory=True)
    
    # Initialisation du modèle, en gardant la configuration de gel des paramètres
    model = TemporalPlanetDeepLab(num_classes=num_classes).to(device)
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.convlstm.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    
    # Encapsulation dans DDP
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scaler = GradScaler()
    
    # Définition des étapes pour débloquer certains modules
    unfreeze = {"convlstm": 5, "encoder": 10}
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        
        if epoch == unfreeze['convlstm']:
            for p in model.module.convlstm.parameters():
                p.requires_grad = True
            optimizer.add_param_group({"params": model.module.convlstm.parameters(), "lr": lr})
        if epoch == unfreeze['encoder']:
            for p in model.module.encoder.parameters():
                p.requires_grad = True
            optimizer.add_param_group({"params": model.module.encoder.parameters(), "lr": lr * 0.1})
        
        for batch in tqdm(train_loader, desc=f"Fold {fold} - Train Epoch {epoch}/{num_epochs}"):
            X = batch['X'].to(device)
            Y = batch['Y'].to(device)
            mask = batch['mask_superv'].to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                preds = model(X)
                if preds.shape[0] != mask.shape[1]:
                    preds = preds.permute(1, 0, 2, 3, 4)
                loss = temporal_semi_supervised_loss(preds,
                                                     {"Y": Y, "mask_superv": mask},
                                                    lambda_temp=0.1, 
                                                    lambda_tversky=1.0,
                                                    lambda_lovasz=1.0, 
                                                    num_classes=num_classes,
                                                    class_weights = class_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        if local_rank == 0:
            print(f"Fold {fold} - Epoch {epoch} avg loss: {avg_loss:.4f}")
            evaluator = MetricsEvaluator(num_classes, ignore_index=255)
            val_metrics = evaluate(model.module, val_loader, evaluator, device)
            display_metrics(val_metrics, epoch, fold)
            
            # Sauvegarde du meilleur modèle du pli
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                torch.save(model.state_dict(), f"best_model_fold_{fold}.pth")
    
    return best_val_loss

def main():
    # Initialisation du groupe de processus distribué (DDP)
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Hyperparamètres et chemins
    global CSV_PATH, ALIGNMENT_CSV, SEQ_LENGTH
    CSV_PATH = "dataframe/df_merged_expanded.csv"
    ALIGNMENT_CSV = "dataframe/keyframes_alignment_geotorch.csv"
    SEQ_LENGTH = 12
    batch_size = 2 * 2
    num_workers = 20
    num_epochs = 60
    num_classes = 7
    lr = 1e-4
    n_splits = 5
    

    
    # Chargement du CSV complet
    df_full = pd.read_csv(CSV_PATH)
    groups = df_full["key"]
    gkf = GroupKFold(n_splits=n_splits)
    
    os.makedirs("dataframe", exist_ok=True)
    
    # Création des fichiers CSV pour chaque pli uniquement par le processus de rang 0
    if local_rank == 0:
        for fold, (train_idx, val_idx) in enumerate(gkf.split(df_full, groups=groups)):
            train_df = df_full.iloc[train_idx]
            val_df = df_full.iloc[val_idx]
            train_csv = f"dataframe/train_fold_{fold}.csv"
            val_csv = f"dataframe/val_fold_{fold}.csv"
            train_df.to_csv(train_csv, index=False)
            val_df.to_csv(val_csv, index=False)
    # Synchronisation de tous les processus pour s'assurer que les CSV sont créés
    dist.barrier()
    
    best_models = []
    fold = 0
    # Chaque processus lit les CSV existants pour chaque pli
    for _, _ in gkf.split(df_full, groups=groups):
        train_csv = f"dataframe/train_fold_{fold}.csv"
        val_csv = f"dataframe/val_fold_{fold}.csv"
        if local_rank == 0:
            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv)
            print(f"=== Démarrage du Fold {fold} ===")
            print(f"Taille du set d'entraînement: {len(train_df.groupby('key'))}")
            print(f"Taille du set de validation: {len(val_df.groupby('key'))}")
        best_val_loss = train_fold(train_csv, val_csv, fold, device, local_rank,
                                    num_epochs, batch_size, num_workers, num_classes, lr)
        best_models.append((fold, best_val_loss))
        fold += 1
    
    if local_rank == 0:
        print("Résultats des meilleurs modèles par fold :", best_models)
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
