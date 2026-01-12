# ==============================================================================
# model.R
# ==============================================================================
library(reticulate)

use_condaenv("r-amortised", required = TRUE)

torch <- import("torch")
nn    <- torch$nn

py_run_string("
import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Python using device: {device}')

class VectorField(nn.Module):
    def __init__(self, context_dim=40):
        super().__init__()
        input_dim = 2 + 1 + context_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.Softplus(),
            nn.Linear(256, 256), nn.Softplus(),
            nn.Linear(256, 256), nn.Softplus(),
            nn.Linear(256, 2)
        )

    def forward(self, t, theta, context):
        if not torch.is_tensor(t) or t.ndim == 0:
            t = torch.ones(theta.shape[0], 1).to(theta.device) * t
        elif t.ndim == 1:
            t = t.unsqueeze(1)
        x = torch.cat([theta, t, context], dim=-1)
        return self.net(x)

@torch.no_grad()
def sample_batch_ode(model, contexts, steps=50):
    model.eval()
    batch_size = contexts.shape[0]
    x_t = torch.randn(batch_size, 2).to(device)
    dt = 1.0 / steps
    for i in range(steps):
        t_val = i * dt
        v = model(t_val, x_t, contexts)
        x_t = x_t + v * dt
    return x_t.cpu().numpy()

@torch.no_grad()
def get_ode_trajectory(model, context_obs, steps=100, save_every=5):
    model.eval()
    dt = 1.0 / steps
    n_samples = context_obs.shape[0]
    x_t = torch.randn(n_samples, 2).to(device)

    trajectory = [x_t.cpu().numpy()]
    times = [0.0]

    for i in range(steps):
        t_val = i * dt
        t_tensor = torch.ones(n_samples, 1).to(device) * t_val
        v = model(t_tensor, x_t, context_obs)
        x_t = x_t + v * dt

        if (i + 1) % save_every == 0:
            trajectory.append(x_t.cpu().numpy())
            times.append(t_val + dt)

    return trajectory, times
")
