import sys, os
sys.path.append(os.path.abspath("src"))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader_sat_image import TemporalSatDataset, default_augmentation_fn, validation_fn
from loss_sat_images import temporal_semi_supervised_loss
from model_deeplabv3_plus import TemporalPlanetDeepLab
from metrics import MetricsEvaluator  # Assurez-vous que le module metrics.py est dans src/

def print_debug_info(batch, preds, loss):
    X, Y, sup_mask = batch["X"], batch["Y"], batch["mask_superv"]
    print(f"[DEBUG] Batch shapes -> X: {X.shape}, Y: {Y.shape}, mask_superv: {sup_mask.shape}")
    print(f"[DEBUG] Preds shape: {preds.shape}, Loss: {loss.item():.4f}")

def evaluate(model, loader, evaluator, device):
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluation", unit="batch")):
            
            X = batch["X"].to(device)    # [B, T, 4, H, W]
            Y = batch["Y"].to(device)    # [B, T, H, W]
            preds = model(X)  # Peut être [T, B, C, H, W] ou [B, T, C, H, W]
            if preds.shape[0] != batch["mask_superv"].shape[1]:
                preds = preds.permute(1, 0, 2, 3, 4)
            
            T, B, C, H, W = preds.shape

            for t in range(T):
                for b in range(B):
                    pred_mask = torch.argmax(preds[t, b], dim=0)  # [H, W]
                    gt_mask = Y[b, t]  # [H, W]
                    evaluator.update(pred_mask, gt_mask)
    metrics = evaluator.compute()
    return metrics

# Chemins et hyperparamètres
CSV_PATH      = "dataframe/df_merged_expanded.csv"
ALIGNMENT_CSV = "dataframe/keyframes_alignment_geotorch.csv"
BATCH_SIZE    = 8
NUM_WORKERS   = 0
SEQ_LENGTH    = 16
NUM_CLASSES   = 8
LR            = 1e-4
NUM_EPOCHS    = 60
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Création des ensembles avec split réalisé dans le code
# Pour l'entraînement, on garde l'augmentation par défaut
train_dataset = TemporalSatDataset(csv_path=CSV_PATH,
                                     alignment_csv=ALIGNMENT_CSV,
                                     transform_fn=default_augmentation_fn,
                                     seq_length=SEQ_LENGTH,
                                     random_subseq=True,
                                     split='train')
# Pour validation et test, on désactive les augmentations aléatoires
val_dataset = TemporalSatDataset(csv_path=CSV_PATH,
                                   alignment_csv=ALIGNMENT_CSV,
                                   transform_fn=validation_fn,
                                   seq_length=SEQ_LENGTH*4,
                                   random_subseq=False,
                                   split='val')
test_dataset = TemporalSatDataset(csv_path=CSV_PATH,
                                    alignment_csv=ALIGNMENT_CSV,
                                    transform_fn=validation_fn,
                                    seq_length=SEQ_LENGTH*4,
                                    random_subseq=False,
                                    split='test')

# Debug : afficher la taille des datasets
print(f"[DEBUG] Train dataset length: {len(train_dataset)}")
print(f"[DEBUG] Validation dataset length: {len(val_dataset)}")
print(f"[DEBUG] Test dataset length: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

# Debug : afficher le nombre de batches dans chaque DataLoader
print(f"[DEBUG] Train loader batches: {len(train_loader)}")
print(f"[DEBUG] Validation loader batches: {len(val_loader)}")
print(f"[DEBUG] Test loader batches: {len(test_loader)}")

# Initialisation du modèle et de l'optimiseur
model = TemporalPlanetDeepLab(num_classes=NUM_CLASSES).to(DEVICE)
# Au départ, on gèle l'encodeur et le convlstm,
# mais on dé-gèle dès le début la tête de classification.
for param in model.encoder.parameters():
    param.requires_grad = False
for param in model.convlstm.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
# Déblocage progressif : convlstm à l'epoch 15, encoder à l'epoch 30
unfreeze_epochs = {
    "convlstm": 15,
    "encoder": 30
}

print("Starting training")
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    
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
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", unit="batch")
    for batch in loop:
        X = batch["X"].to(DEVICE)    # [B, T, 4, H, W]
        Y = batch["Y"].to(DEVICE)    # [B, T, H, W]
        sup_mask = batch["mask_superv"].to(DEVICE)  # [B, T]

        preds = model(X)
        if preds.shape[0] != batch["mask_superv"].shape[1]:
            preds = preds.permute(1, 0, 2, 3, 4)
        loss = temporal_semi_supervised_loss(
            preds,
            {"Y": Y, "mask_superv": sup_mask},
            lambda_temp=1.0, lambda_dice=1.0, lambda_focal=0.0,
            num_classes=NUM_CLASSES
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / (loop.n + 1))
        
    # Évaluation sur le jeu de validation
    evaluator_val = MetricsEvaluator(num_classes=NUM_CLASSES, ignore_index=255)
    metrics_val = evaluate(model, val_loader, evaluator_val, DEVICE)
    # Évaluation sur le jeu d'entraînement
    evaluator_train = MetricsEvaluator(num_classes=NUM_CLASSES, ignore_index=255)
    metrics_train = evaluate(model, train_loader, evaluator_train, DEVICE)
    
    print(f"[VAL] Epoch {epoch} metrics: {metrics_val}")
    print(f"[TRAIN] Epoch {epoch} metrics: {metrics_train}")

print("Training finished, saving model")
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/temporal_planet_deeplab_progressive.pth")
print("Model saved to models/temporal_planet_deeplab_progressive.pth")

# Évaluation finale sur l'ensemble de test et d'entraînement
evaluator_train_final = MetricsEvaluator(num_classes=NUM_CLASSES, ignore_index=255)
metrics_train_final = evaluate(model, train_loader, evaluator_train_final, DEVICE)
print("[TRAIN] Final metrics:", metrics_train_final)

evaluator_test_final = MetricsEvaluator(num_classes=NUM_CLASSES, ignore_index=255)
metrics_test_final = evaluate(model, test_loader, evaluator_test_final, DEVICE)
print("[TEST] Final metrics:", metrics_test_final)
