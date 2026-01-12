library(reticulate)

use_condaenv("r-amortised", required = TRUE)

torch <- import("torch")
nn    <- torch$nn

py_run_string("
import torch
import torch.nn as nn

class TransformerNet(nn.Module):
    def __init__(self, in_dim, d_model=512, nhead=4, layers=4, out_dim=100):
        super().__init__()
        self.embed = nn.Linear(int(in_dim), d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)

        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, int(out_dim))
        )

        self.sparse_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, int(out_dim)),
            nn.Sigmoid()
        )

    def forward(self, x_list):
        device = next(self.parameters()).device
        r, s = [], []
        for x in x_list:
            x = x.to(device)
            z = self.encoder(self.embed(x).unsqueeze(0)).mean(dim=1)
            r.append(self.reg_head(z))
            s.append(self.sparse_head(z))
        if not r:
            return torch.empty(0, device=device), torch.empty(0, device=device)
        return torch.vstack(r), torch.vstack(s)
")
