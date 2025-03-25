import os
import sys
from sklearn.model_selection import GroupKFold
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from dataloader_sat_image import TemporalSatDataset, default_augmentation_fn, validation_fn
from loss_sat_images import temporal_semi_supervised_loss
from model_deeplabv3_plus import TemporalPlanetDeepLab
from metrics import MetricsEvaluator

# Distributed init
dist.init_process_group(backend="nccl")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# Hyperparams
CSV_PATH      = "dataframe/df_merged_expanded.csv"
ALIGNMENT_CSV = "dataframe/keyframes_alignment_geotorch.csv"
BATCH_SIZE    = 8
NUM_WORKERS   = 20
SEQ_LENGTH    = 16
NUM_CLASSES   = 8
LR            = 1e-4
NUM_EPOCHS    = 60
N_SPLITS      = 5

df = pd.read_csv(CSV_PATH)
groups = df["key"].values

gkf = GroupKFold(n_splits=N_SPLITS)

def train_and_validate(train_df, val_df):
    train_ds = TemporalSatDataset(train_df, ALIGNMENT_CSV, default_augmentation_fn, seq_length=SEQ_LENGTH, random_subseq=True, split='train')
    val_ds   = TemporalSatDataset(val_df, ALIGNMENT_CSV, validation_fn, seq_length=SEQ_LENGTH*4, random_subseq=False, split='val')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=DistributedSampler(train_ds), num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, sampler=DistributedSampler(val_ds, shuffle=False), num_workers=NUM_WORKERS, pin_memory=True)

    model = TemporalPlanetDeepLab(num_classes=NUM_CLASSES).to(device)
    for p in model.encoder.parameters(): p.requires_grad=False
    for p in model.convlstm.parameters(): p.requires_grad=False
    for p in model.classifier.parameters(): p.requires_grad=True

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scaler = GradScaler()

    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        running_loss = 0.0

        optimizer.zero_grad()
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}/{NUM_EPOCHS}", disable=(local_rank!=0)):
            X = batch["X"].to(device)
            Y = batch["Y"].to(device)
            mask = batch["mask_superv"].to(device)

            with autocast():
                preds = model(X)
                if preds.shape[0] != mask.shape[1]:
                    preds = preds.permute(1,0,2,3,4)
                loss = temporal_semi_supervised_loss(preds, {"Y": Y, "mask_superv": mask},
                                                     lambda_temp=1.0, lambda_dice=1.0,
                                                     lambda_focal=0.0, num_classes=NUM_CLASSES)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        if local_rank==0:
            print(f"Epoch {epoch} avg loss: {running_loss/len(train_loader):.4f}")

    if local_rank==0:
        val_metrics = evaluate(model.module, val_loader, MetricsEvaluator(NUM_CLASSES, ignore_index=255), device)
        print(f"Validation metrics: {val_metrics}")
    dist.barrier()
    return model.module

def evaluate(model, loader, evaluator, device):
    model.eval()
    with torch.no_grad(), autocast():
        for batch in tqdm(loader, desc="Evaluation", disable=(local_rank!=0)):
            X = batch["X"].to(device); Y = batch["Y"].to(device)
            preds = model(X)
            if preds.shape[0] != batch["mask_superv"].shape[1]:
                preds = preds.permute(1,0,2,3,4)
            T,B,_,H,W = preds.shape
            for t in range(T):
                for b in range(B):
                    evaluator.update(preds[t,b].argmax(0), Y[b,t])
    return evaluator.compute()

# Cross‑validation loop
best_models = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups), start=1):
    if local_rank==0:
        print(f"\n===== Fold {fold}/{N_SPLITS} =====")
    train_df = df.iloc[train_idx]; val_df = df.iloc[val_idx]
    best_models.append(train_and_validate(train_df, val_df))

if local_rank==0:
    dist.destroy_process_group()
    torch.cuda.empty_cache()
