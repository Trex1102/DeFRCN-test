import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from .cond_residual_vae import CondResidualVAE
from .vae_loss import vae_loss_function

# ==========================================
#                 CONFIG
# ==========================================
FEAT_FILES = [
    "data/roifeats_base/voc_2007_trainval_base1_combined.npz",
    "data/roifeats_base/voc_2012_trainval_base1_combined.npz"
]
CLIP_FILE = "data/roifeats_base/clip_prototypes.npy"
OUTPUT_MODEL_PATH = "checkpoints/defrcn_vae_model.pth"

# Training Hyperparameters
BATCH_SIZE = 64
LR = 5e-5           # Lower LR for stability with large model
EPOCHS = 200        # Increased to 200 as requested
PATIENCE = 20       # Stop if no improvement for 20 epochs
MIN_DELTA = 0.1     # Minimum change to count as improvement

# Model Architecture Dimensions
RESID_DIM = 2048
SEM_DIM = 512
LATENT_DIM = 512
HIDDEN_DIM = 4096

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)

# ==========================================
#              DATASET CLASS
# ==========================================
class CombinedFeatureDataset(Dataset):
    def __init__(self, feat_files, clip_path):
        all_feats = []
        all_classes = []

        print(f"Loading CLIP prototypes from {clip_path}...")
        self.clip_prototypes = torch.from_numpy(np.load(clip_path)).float()

        print("Loading Feature Files:")
        for fpath in feat_files:
            if not os.path.exists(fpath):
                print(f"  [WARN] File {fpath} not found. Skipping.")
                continue
            print(f"  -> Reading {fpath} ...")
            data = np.load(fpath)
            all_feats.append(data['feats'])
            all_classes.append(data['classes'])

        if not all_feats:
            raise RuntimeError("No feature files loaded!")

        combined_feats = np.concatenate(all_feats, axis=0)
        combined_classes = np.concatenate(all_classes, axis=0)

        self.feats = torch.from_numpy(combined_feats).float()
        self.classes = torch.from_numpy(combined_classes).long()

        print("Calculating Global Normalization Stats...")
        self.feat_mean = self.feats.mean(dim=0)
        self.feat_std = self.feats.std(dim=0) + 1e-6
        self.feats = (self.feats - self.feat_mean) / self.feat_std

        print(f"Dataset Ready. Total Samples: {len(self.feats)}")

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        visual_feat = self.feats[idx]
        cls_id = self.classes[idx]
        
        if cls_id < 0 or cls_id >= len(self.clip_prototypes):
            text_feat = torch.zeros(self.clip_prototypes.shape[1])
        else:
            text_feat = self.clip_prototypes[cls_id]
            
        return visual_feat, text_feat

# ==========================================
#          ROBUST LOSS FUNCTION
# ==========================================



# ==========================================
#              MAIN LOOP
# ==========================================
def main():
    print("Initializing Training with Early Stopping...")
    
    dataset = CombinedFeatureDataset(FEAT_FILES, CLIP_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = CondResidualVAE(
        resid_dim=RESID_DIM, sem_dim=SEM_DIM, latent_dim=LATENT_DIM, hidden_h=HIDDEN_DIM
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()

    # --- Early Stopping Variables ---
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_mse = 0
        total_kld = 0
        
        for batch_idx, (visual_feat, clip_feat) in enumerate(dataloader):
            visual_feat = visual_feat.to(device)
            clip_feat = clip_feat.to(device)
            
            optimizer.zero_grad()
            
            recon, mu_q, logvar_q, mu_p, logvar_p, z = model(visual_feat, clip_feat)
            loss, mse, kld = vae_loss_function(recon, visual_feat, mu_q, logvar_q, mu_p, logvar_p)
            
            if torch.isnan(loss):
                print("Error: NaN loss detected. Stopping.")
                return

            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_kld += kld.item()

        avg_loss = total_loss / len(dataset)
        avg_mse = total_mse / len(dataset)
        avg_kld = total_kld / len(dataset)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.1f} (MSE: {avg_mse:.1f} | KLD: {avg_kld:.1f})")

        # --- Early Stopping Logic ---
        if avg_loss < (best_loss - MIN_DELTA):
            best_loss = avg_loss
            patience_counter = 0
            
            # Save the BEST model
            torch.save({
                'model_state_dict': model.state_dict(),
                'feat_mean': dataset.feat_mean,
                'feat_std': dataset.feat_std,
                'epoch': epoch
            }, OUTPUT_MODEL_PATH)
            print(f"  --> Improvement! Model saved. (Patience Reset)")
        else:
            patience_counter += 1
            print(f"  --> No improvement. Patience: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print("\nEarly Stopping Triggered! Training finished.")
                break

    print(f"Training Complete. Best Loss: {best_loss:.2f}")

if __name__ == "__main__":
    main()