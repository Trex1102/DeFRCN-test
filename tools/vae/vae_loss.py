import torch
import torch.nn as nn

def vae_loss_function(recon, target, mu_q, logvar_q, mu_p, logvar_p):
    mse = nn.functional.mse_loss(recon, target, reduction='sum')
    
    # Robust KLD Calculation (Prevents Nan)
    logvar_q = torch.clamp(logvar_q, min=-10, max=10)
    logvar_p = torch.clamp(logvar_p, min=-10, max=10)
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    
    kl_element = 0.5 * ( (var_q + (mu_q - mu_p).pow(2)) / (var_p + 1e-6) - 1 + (logvar_p - logvar_q) )
    kld = torch.sum(kl_element)

    return mse + kld, mse, kld