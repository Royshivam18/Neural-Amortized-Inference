# ==============================================================================
# trainer.R
# ==============================================================================

library(reticulate)
library(ggplot2)
library(dplyr)
library(tidyr)
library(viridis)

source("Neural_Amortised_Sampler/data_generator.R")
source("Neural_Amortised_Sampler/model.R")

use_condaenv("r-amortised", required = TRUE)

torch <- import("torch")
nn    <- torch$nn

# ----------------------------
# Settings
# ----------------------------
POINTS_PER_TASK <- 20L
N_TRAIN <- 5000L

# ----------------------------
# Generate training data
# ----------------------------
tasks <- lapply(1:N_TRAIN, function(i) generate_8mode_task(n_points = POINTS_PER_TASK))
train_contexts <- do.call(rbind, lapply(tasks, function(t) t$context_flat))
train_params   <- do.call(rbind, lapply(tasks, function(t) t$params))

# Visualization target (optional; used only if you keep viz in trainer)
vis_idx <- 1
vis_context_R   <- train_contexts[vis_idx, , drop=FALSE]
vis_true_beta   <- train_params[vis_idx, ]
vis_context_tensor <- torch$tensor(vis_context_R, dtype=torch$float32)$to(py$device)

# ----------------------------
# Trainer
# ----------------------------
train_flow_matching_with_viz <- function(
    model,
    contexts,
    parameters,
    vis_ctx_tensor,
    vis_true_prm,
    epochs = 100L,
    batch_size = 128L,
    lr = 1e-3
) {
  opt <- torch$optim$Adam(model$parameters(), lr = lr)
  n_samples <- nrow(contexts)
  batch_size <- as.integer(batch_size)
  
  for (e in 1:as.integer(epochs)) {
    model$train()
    perm_idx <- sample(n_samples)
    epoch_loss <- 0
    steps <- 0
    
    for (s in seq(1, n_samples, by = batch_size)) {
      end_idx <- min(s + batch_size - 1L, n_samples)
      idx <- perm_idx[s:end_idx]
      current_bs <- length(idx)
      
      ctx_batch <- torch$tensor(contexts[idx, , drop=FALSE], dtype=torch$float32)$to(py$device)
      x_1 <- torch$tensor(parameters[idx, , drop=FALSE], dtype=torch$float32)$to(py$device)
      
      x_0 <- torch$randn_like(x_1)$to(py$device)
      t   <- torch$rand(as.integer(current_bs), 1L)$to(py$device)
      x_t <- (1 - t) * x_0 + t * x_1
      
      target_vector <- x_1 - x_0
      pred_vector   <- model(t, x_t, ctx_batch)
      
      loss <- torch$mean((pred_vector - target_vector)^2)
      
      opt$zero_grad()
      loss$backward()
      opt$step()
      
      epoch_loss <- epoch_loss + loss$item()
      steps <- steps + 1
    }
    
    if (e %% 5L == 0L || e == 1L) {
      cat(sprintf("Epoch %3d | FM Loss: %.5f\n", e, epoch_loss / steps))
      # If you want, plug your progress plot saving here.
    }
  }
  
  invisible(model)
}

# ----------------------------
# Instantiate model + train
# ----------------------------
model <- py$VectorField(context_dim = as.integer(POINTS_PER_TASK * 2))
model$to(py$device)

train_flow_matching_with_viz(
  model,
  train_contexts,
  train_params,
  vis_context_tensor,
  vis_true_beta,
  epochs = 150L,
  batch_size = 256L,
  lr = 1e-3
)
