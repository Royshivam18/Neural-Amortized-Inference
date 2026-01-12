# ==============================================================================
# FILE 1: models.R  (Python models via reticulate)
# ==============================================================================
library(reticulate)

use_condaenv("r-amortised", required = TRUE)

torch <- import("torch")
nn    <- import("torch.nn")

py_run_string("
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerNet(nn.Module):
    def __init__(self, in_dim, d_model=128, nhead=4, layers=2, out_dim=6):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_dim)
        )

    def forward(self, x_list):
        outs = []
        for x in x_list:
            x = x.to(device)
            h = self.embed(x).unsqueeze(0)   # [1, L, D]
            h = self.encoder(h)             # [1, L, D]
            h = h.mean(dim=1)               # [1, D]
            outs.append(self.head(h))       # [1, out_dim]
        return torch.cat(outs, dim=0)

class DeepSetsNet(nn.Module):
    def __init__(self, in_dim, h_dim=128, out_dim=6):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU()
        )
        self.rho = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, out_dim)
        )

    def forward(self, x_list):
        outs = []
        for x in x_list:
            x = x.to(device)
            h = self.phi(x)                 # [L, h_dim]
            h = h.mean(dim=0, keepdim=True) # [1, h_dim]
            outs.append(self.rho(h))        # [1, out_dim]
        return torch.cat(outs, dim=0)
")
