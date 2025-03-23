import sys, os
sys.path.append(os.path.abspath("src"))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader_sat_image import TemporalSatDataset, default_augmentation_fn
from loss_sat_images import temporal_semi_supervised_loss
from model_deeplabv3_plus import TemporalPlanetDeepLab

# Paths & hyperparameters
CSV_PATH      = "dataframe/df_merged_expanded.csv"
ALIGNMENT_CSV = "dataframe/keyframes_alignment_geotorch.csv"
BATCH_SIZE    = 8
NUM_WORKERS   = 20
SEQ_LENGTH    = 16
NUM_CLASSES   = 8
LR            = 1e-4
NUM_EPOCHS    = 60
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset + DataLoader
dataset = TemporalSatDataset(
    csv_path=CSV_PATH,
    alignment_csv=ALIGNMENT_CSV,
    transform_fn=default_augmentation_fn,
    seq_length=SEQ_LENGTH,
    random_subseq=True
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=NUM_WORKERS, pin_memory=True)

# Model + optimizer
model = TemporalPlanetDeepLab(num_classes=NUM_CLASSES).to(DEVICE)
for param in model.encoder.parameters():
    param.requires_grad = False
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

unfreeze_epochs = [15, 30, 45]

print("Starting training")
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0

    # Progressive unfreeze
    if epoch == unfreeze_epochs[0]:
        print("Unfreezing classifier head")
        for p in model.classifier.parameters(): p.requires_grad = True
        optimizer.add_param_group({"params": model.classifier.parameters(), "lr": LR})

    if epoch == unfreeze_epochs[1]:
        print("Unfreezing ConvLSTM")
        for p in model.convlstm.parameters(): p.requires_grad = True
        optimizer.add_param_group({"params": model.convlstm.parameters(), "lr": LR})

    if epoch == unfreeze_epochs[2]:
        print("Unfreezing full backbone")
        for p in model.encoder.parameters(): p.requires_grad = True
        optimizer.add_param_group({"params": model.encoder.parameters(), "lr": LR * 0.1})

    loop = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", unit="batch")
    for batch in loop:
        X = batch["X"].to(DEVICE)
        Y = batch["Y"].to(DEVICE)
        sup_mask = batch["mask_superv"].to(DEVICE)

        preds = model(X)
        loss = temporal_semi_supervised_loss(
            preds.unsqueeze(0),
            {"Y": Y[:, -1].unsqueeze(0), "mask_superv": sup_mask[:, -1].unsqueeze(0)},
            lambda_temp=1.0, lambda_dice=1.0, lambda_focal=0.0, num_classes=NUM_CLASSES
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / (loop.n + 1))

    print(f"Epoch {epoch} completed — Avg Loss: {running_loss/len(loader):.4f}")

print("Training finished, saving model")
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/temporal_planet_deeplab_progressive.pth")
print("Model saved to models/temporal_planet_deeplab_progressive.pth")
