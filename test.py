import sys, os
sys.path.append(os.path.abspath("src"))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader_sat_image import TemporalSatDataset, default_augmentation_fn
from loss_sat_images import temporal_semi_supervised_loss
from model_deeplabv3_plus import TemporalPlanetDeepLab

def print_debug_info(batch, preds, loss):
    X, Y, sup_mask = batch["X"], batch["Y"], batch["mask_superv"]
    print(f"[DEBUG] Batch shapes -> X: {X.shape}, Y: {Y.shape}, mask_superv: {sup_mask.shape}")
    print(f"[DEBUG] Preds shape: {preds.shape}, Loss: {loss.item():.4f}")

# Paths & hyperparamètres
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
# Au départ, on gèle l'encodeur et le convlstm,
# MAIS on dé-gèle dès le début la tête de classification.
for param in model.encoder.parameters():
    param.requires_grad = False
for param in model.convlstm.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True  # Dé-gel immédiat de la tête de classification

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
# Déblocage progressif prévu :
unfreeze_epochs = {
    "convlstm": 15,
    "encoder": 30
}

print("Starting training")
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    print(f"[DEBUG] Epoch {epoch} start")
    
    # Déblocage progressif pour convlstm et encoder
    if epoch == unfreeze_epochs["convlstm"]:
        print("[DEBUG] Unfreezing ConvLSTM")
        for p in model.convlstm.parameters():
            p.requires_grad = True
        optimizer.add_param_group({"params": model.convlstm.parameters(), "lr": LR})
    if epoch == unfreeze_epochs["encoder"]:
        print("[DEBUG] Unfreezing full backbone (encoder)")
        for p in model.encoder.parameters():
            p.requires_grad = True
        optimizer.add_param_group({"params": model.encoder.parameters(), "lr": LR * 0.1})
    
    loop = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", unit="batch")
    for batch in loop:
        # Debug: nombre de frames annotées dans le batch
        num_annotated = batch["mask_superv"].sum().item()
        total_frames = batch["mask_superv"].numel()
        print(f"[DEBUG] Batch contains {num_annotated} annotated frames out of {total_frames} total frames.")

        X = batch["X"].to(DEVICE)    # [B, T, 4, H, W]
        Y = batch["Y"].to(DEVICE)    # [B, T, H, W]
        sup_mask = batch["mask_superv"].to(DEVICE)  # [B, T]

        # Forward pass (le modèle renvoie des logits de forme [T, B, C, H, W] ou [B, T, C, H, W])
        preds = model(X)
        
        loss = temporal_semi_supervised_loss(
            preds,
            {"Y": Y, "mask_superv": sup_mask},
            lambda_temp=1.0, lambda_dice=1.0, lambda_focal=0.0,
            num_classes=NUM_CLASSES
        )

        if loss.grad_fn is None:
            print("[WARNING] Loss is not differentiable!")
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / (loop.n + 1))
        print_debug_info(batch, preds, loss)

    print(f"[DEBUG] Epoch {epoch} completed — Avg Loss: {running_loss/len(loader):.4f}")

print("Training finished, saving model")
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/temporal_planet_deeplab_progressive.pth")
print("Model saved to models/temporal_planet_deeplab_progressive.pth")
