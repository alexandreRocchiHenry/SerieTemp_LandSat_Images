import os
import sys
sys.path.append(os.path.abspath("src"))

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

# Initialize distributed process group
dist.init_process_group(backend="nccl")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

def evaluate(model, loader, evaluator, device):
    model.eval()
    with torch.no_grad(), autocast():
        for batch in tqdm(loader, desc="Evaluation", unit="batch"):
            X = batch["X"].to(device)
            Y = batch["Y"].to(device)
            preds = model(X)
            if preds.shape[0] != batch["mask_superv"].shape[1]:
                preds = preds.permute(1, 0, 2, 3, 4)

            T, B, C, H, W = preds.shape
            for t in range(T):
                for b in range(B):
                    pred_mask = torch.argmax(preds[t, b], dim=0)
                    evaluator.update(pred_mask, Y[b, t])
    return evaluator.compute()

# Hyperparameters
CSV_PATH = "dataframe/df_merged_expanded.csv"
ALIGNMENT_CSV = "dataframe/keyframes_alignment_geotorch.csv"
BATCH_SIZE = 8
NUM_WORKERS = 20
SEQ_LENGTH = 16
NUM_CLASSES = 8
LR = 1e-4
NUM_EPOCHS = 60

# Datasets + Samplers + Loaders
train_dataset = TemporalSatDataset(CSV_PATH, ALIGNMENT_CSV, default_augmentation_fn, seq_length=SEQ_LENGTH, random_subseq=True, split='train')
val_dataset   = TemporalSatDataset(CSV_PATH, ALIGNMENT_CSV, validation_fn, seq_length=SEQ_LENGTH*4, random_subseq=False, split='val')
test_dataset  = TemporalSatDataset(CSV_PATH, ALIGNMENT_CSV, validation_fn, seq_length=SEQ_LENGTH*4, random_subseq=False, split='test')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=DistributedSampler(train_dataset), num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, sampler=DistributedSampler(val_dataset, shuffle=False), num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, sampler=DistributedSampler(test_dataset, shuffle=False), num_workers=NUM_WORKERS, pin_memory=True)

# Model + optimizer + scaler
model = TemporalPlanetDeepLab(num_classes=NUM_CLASSES).to(device)
for p in model.encoder.parameters(): p.requires_grad = False
for p in model.convlstm.parameters(): p.requires_grad = False
for p in model.classifier.parameters(): p.requires_grad = True

model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scaler = GradScaler()

unfreeze = {"convlstm": 15, "encoder": 30}

# Training loop
for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    train_loader.sampler.set_epoch(epoch)
    running_loss = 0.0

    if epoch == unfreeze['convlstm']:
        for p in model.module.convlstm.parameters(): p.requires_grad = True
        optimizer.add_param_group({"params": model.module.convlstm.parameters(), "lr": LR})
    if epoch == unfreeze['encoder']:
        for p in model.module.encoder.parameters(): p.requires_grad = True
        optimizer.add_param_group({"params": model.module.encoder.parameters(), "lr": LR * 0.1})

    for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}/{NUM_EPOCHS}"):
        X, Y, mask = batch['X'].to(device), batch['Y'].to(device), batch['mask_superv'].to(device)

        optimizer.zero_grad()
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

    if local_rank == 0:
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")
        val_metrics = evaluate(model.module, val_loader, MetricsEvaluator(NUM_CLASSES, ignore_index=255), device)
        print(f"Validation metrics: {val_metrics}")
        
if local_rank == 0:
    torch.cuda.empty_cache()
    dist.destroy_process_group()


# Final evaluation on rank0
if local_rank == 0:
    test_metrics = evaluate(model.module, test_loader, MetricsEvaluator(NUM_CLASSES, ignore_index=255), device)
    print(f"Test metrics: {test_metrics}")
