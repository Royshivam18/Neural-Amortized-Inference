# ==============================================================================
# FILE 3: train.R  (trainer for amortized beta prediction)
# ==============================================================================
library(reticulate)

source("models.R")
source("generative.R")

use_condaenv("r-amortised", required = TRUE)
torch <- import("torch")
nn    <- import("torch.nn")

train_amortised <- function(
    arch=c("transformer","deepsets"),
    p=20, K=8, tau=3.0,
    sigma=1.0,
    steps=5000L, batch_size=64L,
    lr=1e-3,
    d_model=128L, nhead=4L, layers=2L,
    h_dim=128L,
    seed=1L,
    eval_every=250L
) {
  arch <- match.arg(arch)
  device <- if (torch$cuda$is_available()) "cuda" else "cpu"
  torch$manual_seed(as.integer(seed))
  
  centroids <- init_centroids(K=K, p=p, tau=tau, seed=seed, device=device)
  
  in_dim  <- p + 1L
  out_dim <- p
  
  if (arch == "transformer") {
    model <- py$TransformerNet(in_dim=in_dim, d_model=as.integer(d_model),
                               nhead=as.integer(nhead), layers=as.integer(layers),
                               out_dim=as.integer(out_dim))
  } else {
    model <- py$DeepSetsNet(in_dim=in_dim, h_dim=as.integer(h_dim),
                            out_dim=as.integer(out_dim))
  }
  
  model$to(py$device)
  
  optim <- torch$optim$Adam(model$parameters(), lr=lr)
  loss_fn <- nn$MSELoss()
  
  for (s in seq_len(as.integer(steps))) {
    batch <- sample_task_batch(B=batch_size, centroids=centroids,
                               sigma=sigma, N_min=10L, N_max=30L)
    
    x_list <- batch$x_list
    y_true <- batch$y_beta$to(py$device)
    
    optim$zero_grad()
    y_hat <- model(x_list)
    loss <- loss_fn(y_hat, y_true)
    loss$backward()
    optim$step()
    
    if (s %% as.integer(eval_every) == 0L) {
      loss_val <- as.numeric(loss$item())
      cat(sprintf("step %d | loss %.6f\n", s, loss_val))
    }
  }
  
  list(model=model, centroids=centroids, device=device)
}

evaluate_amortised <- function(model, centroids, n_tasks=1000L, sigma=1.0) {
  loss_fn <- nn$MSELoss(reduction="mean")
  model$eval()
  
  with(torch$no_grad(), {
    batch <- sample_task_batch(B=as.integer(n_tasks), centroids=centroids, sigma=sigma)
    y_hat <- model(batch$x_list)
    y_true <- batch$y_beta
    loss <- loss_fn(y_hat, y_true)
  })
  
  model$train()
  as.numeric(loss$item())
}

# Example:
# fit <- train_amortised(arch="transformer", K=8, steps=3000, batch_size=64, lr=1e-3)
# test_mse <- evaluate_amortised(fit$model, fit$centroids, n_tasks=2000)
# cat("test MSE:", test_mse, "\n")
