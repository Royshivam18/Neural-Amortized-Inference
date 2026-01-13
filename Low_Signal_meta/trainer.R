# ==============================================================================
# train.R  (train TF and DS separately, optionally resume from saved weights)
# ==============================================================================

suppressPackageStartupMessages({
  library(reticulate)
})

# If you use conda env:
# use_condaenv("r-amortised", required = TRUE)

torch <- import("torch")
nn    <- import("torch.nn")

source("Low_Signal_meta/model.R")          # py$TransformerNet, py$DeepSetsNet
source("Low_Signal_meta/data_generator.R") # init_centroids(), sample_task_batch()

# ------------------------------------------------------------------------------
# Device (torch device object)
# ------------------------------------------------------------------------------
device <- if (torch$cuda$is_available()) torch$device("cuda") else torch$device("cpu")
cat(sprintf("\n--- Device: %s ---\n", device$type))

# ------------------------------------------------------------------------------
# One generic trainer for a single model (TF OR DS)
# ------------------------------------------------------------------------------
train_one_model <- function(
    model,
    centroids,
    steps = 5000L,
    batch_size = 64L,
    sigma = 1.0,
    lr = 1e-3,
    eval_every = 250L,
    save_path = "ckpts/model.pth",
    load_path = NULL
) {
  model$to(device)
  optim <- torch$optim$Adam(model$parameters(), lr = lr)
  loss_fn <- nn$MSELoss()
  
  # ---- resume if weights exist ----
  if (!is.null(load_path) && file.exists(load_path)) {
    cat(sprintf("Loading weights: %s\n", load_path))
    state <- torch$load(load_path, map_location = device)
    model$load_state_dict(state)
    cat("Loaded.\n")
  }
  
  model$train()
  
  for (s in seq_len(as.integer(steps))) {
    batch <- sample_task_batch(
      B = as.integer(batch_size),
      centroids = centroids,
      sigma = sigma,
      N_min = 10L, N_max = 30L
    )
    
    x_list <- batch$x_list            # list of [N_t, p+1] torch tensors
    y_true <- batch$y_beta$to(device) # [B, p]
    
    optim$zero_grad()
    y_hat <- model(x_list)
    loss  <- loss_fn(y_hat, y_true)
    loss$backward()
    optim$step()
    
    if (s == 1L || s %% as.integer(eval_every) == 0L) {
      cat(sprintf("step %d | loss %.6f\n", s, as.numeric(loss$item())))
    }
  }
  
  dir.create(dirname(save_path), showWarnings = FALSE, recursive = TRUE)
  torch$save(model$state_dict(), save_path)
  cat(sprintf("Saved: %s\n", save_path))
  
  invisible(model)
}

# ------------------------------------------------------------------------------
# (Optional) evaluation helper
# ------------------------------------------------------------------------------
eval_model <- function(model, centroids, n_tasks = 1000L, sigma = 1.0) {
  loss_fn <- nn$MSELoss(reduction = "mean")
  model$eval()
  
  out <- NULL
  with(torch$no_grad(), {
    batch <- sample_task_batch(B = as.integer(n_tasks), centroids = centroids, sigma = sigma)
    y_hat  <- model(batch$x_list)
    y_true <- batch$y_beta
    out <- as.numeric(loss_fn(y_hat, y_true)$item())
  })
  
  model$train()
  out
}

# ==============================================================================
# SETTINGS
# ==============================================================================
seed <- 1L
torch$manual_seed(as.integer(seed))

p <- 20L
K <- 8L
tau <- 3.0
sigma <- 1.0

steps <- 5000L
batch_size <- 64L
lr_tf <- 1e-3
lr_ds <- 1e-3
eval_every <- 250L

# ------------------------------------------------------------------------------
# Shared centroids (same latent mixture for both models)
# ------------------------------------------------------------------------------
centroids <- init_centroids(K = K, p = p, tau = tau, seed = seed, device = device)

in_dim  <- p + 1L
out_dim <- p

# ==============================================================================
# 1) TRAIN TRANSFORMER (separately)
# ==============================================================================
cat("\n==================== TRAIN TRANSFORMER ====================\n")
model_tf <- py$TransformerNet(
  in_dim  = as.integer(in_dim),
  d_model = 128L,
  nhead   = 4L,
  layers  = 2L,
  out_dim = as.integer(out_dim)
)
model_tf$to(device)

model_tf <- train_one_model(
  model = model_tf,
  centroids = centroids,
  steps = steps,
  batch_size = batch_size,
  sigma = sigma,
  lr = lr_tf,
  eval_every = eval_every,
  load_path = "ckpts/tf_weights.pth",     # set NULL if you don't want resume
  save_path = "ckpts/tf_weights.pth"
)

mse_tf <- eval_model(model_tf, centroids, n_tasks = 2000L, sigma = sigma)
cat(sprintf("Transformer test MSE: %.6f\n", mse_tf))

# ==============================================================================
# 2) TRAIN DEEPSETS (separately)
# ==============================================================================
cat("\n==================== TRAIN DEEPSETS ====================\n")
model_ds <- py$DeepSetsNet(
  in_dim  = as.integer(in_dim),
  h_dim   = 128L,
  out_dim = as.integer(out_dim)
)
model_ds$to(device)

model_ds <- train_one_model(
  model = model_ds,
  centroids = centroids,
  steps = steps,
  batch_size = batch_size,
  sigma = sigma,
  lr = lr_ds,
  eval_every = eval_every,
  load_path = "ckpts/ds_weights.pth",     # set NULL if you don't want resume
  save_path = "ckpts/ds_weights.pth"
)

mse_ds <- eval_model(model_ds, centroids, n_tasks = 2000L, sigma = sigma)
cat(sprintf("DeepSets test MSE: %.6f\n", mse_ds))

# (Optional) save centroids used in this run
dir.create("ckpts", showWarnings = FALSE, recursive = TRUE)
torch$save(centroids, "ckpts/centroids.pth")
cat("Saved centroids: ckpts/centroids.pth\n")
