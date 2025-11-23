import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# ==========================================
#                 CONFIG
# ==========================================
# List of feature files to combine
FEAT_FILES = [
    "data/roifeats_base/voc_2007_trainval_base1_combined.npz",
    "data/roifeats_base/voc_2012_trainval_base1_combined.npz"
]
CLIP_FILE = "data/roifeats_base/clip_prototypes.npy"
OUTPUT_MODEL_PATH = "checkpoints/defrcn_vae_model.pth"

# Hyperparameters
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 50

# Model Architecture Dimensions
RESID_DIM = 2048    # ResNet feature size
SEM_DIM = 512       # CLIP embedding size
LATENT_DIM = 512    # Latent Z size
HIDDEN_DIM = 4096   # MLP hidden size

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

        # Merge arrays
        combined_feats = np.concatenate(all_feats, axis=0)
        combined_classes = np.concatenate(all_classes, axis=0)

        self.feats = torch.from_numpy(combined_feats).float()
        self.classes = torch.from_numpy(combined_classes).long()

        # --- Standard Normalization (Mean/Std) ---
        # Essential for this VAE architecture which uses Identity output
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
        
        # Get CLIP embedding
        if cls_id < 0 or cls_id >= len(self.clip_prototypes):
            # Fallback for background/unknown
            text_feat = torch.zeros(self.clip_prototypes.shape[1])
        else:
            text_feat = self.clip_prototypes[cls_id]
            
        return visual_feat, text_feat

# ==========================================
#           MODEL ARCHITECTURE
# ==========================================
class CondResidualVAE(nn.Module):
    def __init__(self, resid_dim=2048, sem_dim=512, latent_dim=512, hidden_h=4096, leaky_slope=0.2):
        super().__init__()

        # --- ENCODER ---
        # Projects features independently then concatenates
        self.enc_fcx = nn.Linear(resid_dim, hidden_h)
        self.enc_fcy = nn.Linear(sem_dim, hidden_h)

        self.enc_fc1 = nn.Linear(hidden_h*2, hidden_h)
        self.enc_fc2 = nn.Linear(hidden_h, hidden_h)
        self.enc_fc3 = nn.Linear(hidden_h, hidden_h)

        self.enc_mu = nn.Linear(hidden_h, latent_dim)
        self.enc_logvar = nn.Linear(hidden_h, latent_dim)
        self.enc_act = nn.LeakyReLU(leaky_slope, inplace=True)

        # --- PRIOR (CADA-VAE Style) ---
        # Maps semantic vector (CLIP) -> mu_p, logvar_p
        self.prior_fc1 = nn.Linear(sem_dim, hidden_h)
        self.prior_fc2 = nn.Linear(hidden_h, hidden_h)
        self.prior_mu = nn.Linear(hidden_h, latent_dim)
        self.prior_logvar = nn.Linear(hidden_h, latent_dim)
        self.prior_act = nn.LeakyReLU(leaky_slope, inplace=True)

        # --- DECODER ---
        self.dec_fcx = nn.Linear(latent_dim, hidden_h)
        self.dec_fcy = nn.Linear(sem_dim, hidden_h)

        self.dec_fc1 = nn.Linear(hidden_h*2, hidden_h)
        self.dec_out = nn.Linear(hidden_h, resid_dim)

        self.dec_hidden_act = nn.LeakyReLU(leaky_slope, inplace=True)
        self.dec_out_act = nn.Identity()  # Linear output

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, resid, sem):
        # Project both to hidden_h
        x = self.enc_act(self.enc_fcx(resid))
        y = self.enc_act(self.enc_fcy(sem))
        
        # Concatenate
        x = torch.cat([x,y], dim=1)

        x = self.enc_act(self.enc_fc1(x))
        x = self.enc_act(self.enc_fc2(x))
        x = self.enc_act(self.enc_fc3(x))

        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        return mu, logvar

    def prior(self, sem):
        x = self.prior_act(self.prior_fc1(sem))
        x = self.prior_act(self.prior_fc2(x))
        mu = self.prior_mu(x)
        logvar = self.prior_logvar(x)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, sem):
        # FIXED BUG: Passed z to fcx (latent dim) and sem to fcy (sem dim)
        x = self.dec_hidden_act(self.dec_fcx(z))   
        y = self.dec_hidden_act(self.dec_fcy(sem))
        
        x = torch.cat([x,y], dim=1)

        x = self.dec_hidden_act(self.dec_fc1(x))
        x = self.dec_out(x)
        x = self.dec_out_act(x)
        return x

    def forward(self, resid, sem):
        # 1. Posterior Encoder (q)
        mu_q, logvar_q = self.encode(resid, sem)
        z = self.reparam(mu_q, logvar_q)
        
        # 2. Conditional Prior (p)
        mu_p, logvar_p = self.prior(sem)
        
        # 3. Reconstruction
        recon = self.decode(z, sem)
        
        return recon, mu_q, logvar_q, mu_p, logvar_p, z


# ==========================================
#            LOSS & TRAINING
# ==========================================
def vae_loss_function(recon, target, mu_q, logvar_q, mu_p, logvar_p):
    """
    Loss = MSE + KL(Posterior || Prior)
    """
    # 1. Reconstruction Loss (Sum of Squared Errors)
    mse = nn.functional.mse_loss(recon, target, reduction='sum')
    
    # 2. KL Divergence between two Gaussians
    # q(z|x,c) ~ N(mu_q, var_q)
    # p(z|c)   ~ N(mu_p, var_p)
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    
    # Analytical KL term
    # term1 = log(var_p / var_q) -> logvar_p - logvar_q
    # term2 = (var_q + (mu_q - mu_p)^2) / var_p
    kl_element = 0.5 * ( (var_q + (mu_q - mu_p).pow(2)) / var_p - 1 + (logvar_p - logvar_q) )
    kld = torch.sum(kl_element)

    return mse + kld, mse, kld

def main():
    print("Initializing Training...")
    
    # 1. Dataset
    dataset = CombinedFeatureDataset(FEAT_FILES, CLIP_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 2. Model
    model = CondResidualVAE(
        resid_dim=RESID_DIM,
        sem_dim=SEM_DIM,
        latent_dim=LATENT_DIM,
        hidden_h=HIDDEN_DIM
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"Model created. Param count: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    model.train()

    # 3. Training Loop
    for epoch in range(EPOCHS):
        total_loss = 0
        total_mse = 0
        total_kld = 0
        
        for batch_idx, (visual_feat, clip_feat) in enumerate(dataloader):
            visual_feat = visual_feat.to(device)
            clip_feat = clip_feat.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            recon, mu_q, logvar_q, mu_p, logvar_p, z = model(visual_feat, clip_feat)
            
            # Loss
            loss, mse, kld = vae_loss_function(recon, visual_feat, mu_q, logvar_q, mu_p, logvar_p)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_kld += kld.item()

        # Stats per epoch
        avg_loss = total_loss / len(dataset)
        avg_mse = total_mse / len(dataset)
        avg_kld = total_kld / len(dataset)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Total Loss: {avg_loss:.1f} | MSE: {avg_mse:.1f} | KLD: {avg_kld:.1f}")

    # 4. Save Model & Stats
    print(f"Saving model to {OUTPUT_MODEL_PATH}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'feat_mean': dataset.feat_mean, # CRITICAL: Needed for un-normalizing generated features
        'feat_std': dataset.feat_std
    }, OUTPUT_MODEL_PATH)
    
    print("Training Complete.")

if __name__ == "__main__":
    main()