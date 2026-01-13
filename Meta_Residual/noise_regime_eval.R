# regime_stress_test.R  (NO PADDING VERSION)

suppressPackageStartupMessages({
  library(reticulate)
  library(tibble)
  library(dplyr)
})

# ------------------------------------------------------------------
# LOAD MODELS + DATA GENERATOR
# ------------------------------------------------------------------
source("Meta_Residual/model.R")          # provides py$TransformerNet, py$DeepSetsNet
source("Meta_Residual/data_generator.R") # must provide sample_regression_batch()

torch <- import("torch")

# ------------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------------
device <- if (torch$cuda$is_available()) torch$device("cuda") else torch$device("cpu")
cat(sprintf("\n--- Starting Stress Test on Device: %s ---\n", device$type))

# ------------------------------------------------------------------
# INSTANTIATE + LOAD WEIGHTS
# ------------------------------------------------------------------
model_tf <- py$TransformerNet(in_dim = 6L, out_dim = 6L)$to(device)
model_ds <- py$DeepSetsNet(in_dim = 6L, out_dim = 6L)$to(device)

cat("Reloading weights...\n")
model_tf$load_state_dict(torch$load("Meta_Residual/model_res_weights/tf_n400.pth", map_location = device))
model_ds$load_state_dict(torch$load("Meta_Residual/model_res_weights/ds_n400.pth", map_location = device))
cat("Weights loaded.\n")

model_tf$eval()
model_ds$eval()

# ==============================================================================
# 1. Simulation Settings (1 X-dist, 11 Noise Types)
# ==============================================================================
x_configs_simple <- list(
  list(name = "Norm_Std", dist = "normal", params = list(mean = 0, sd = 1))
)

noise_configs <- list(
  list(name = "Clean_Small",     dist = "normal",      params = list(sd = 0.1)),
  list(name = "Clean_High",      dist = "normal",      params = list(sd = 2)),
  list(name = "Right_Skew_r1",   dist = "exponential", params = list(rate = 1.0)),
  list(name = "Left_Skew_r1",    dist = "skew_left",   params = list(rate = 1.0)),
  list(name = "Right_Skew_r2",   dist = "exponential", params = list(rate = 2.0)),
  list(name = "Left_Skew_r2",    dist = "skew_left",   params = list(rate = 2.0)),
  list(name = "Bi_Unbal_m3s1",   dist = "bimodal",     params = list(mu = 3, sd = 1, probs = c(0.8, 0.2))),
  list(name = "Bi_Unbal_m4s2",   dist = "bimodal",     params = list(mu = 4, sd = 2, probs = c(0.9, 0.1))),
  list(name = "TriModal_tight",  dist = "trimodal",    params = list(mu = 4, sd_main = 1, sd_side = 0.1, p_side = 0.1)),
  list(name = "TriModal_wide",   dist = "trimodal",    params = list(mu = 1, sd_main = 1, sd_side = 2,   p_side = 0.2))
)

# ==============================================================================
# 2. Helpers (NO PAD)
# ==============================================================================
# Convert list of sample matrices -> list of torch tensors (Li x d_in), sent to device
to_list_of_tensors <- function(sample_list, device) {
  lapply(sample_list, function(M) {
    torch$tensor(as.matrix(M), dtype = torch$float32)$to(device)
  })
}

# OLS baseline: return per-task MSE on (beta,c)
ols_errors <- function(ds_eval) {
  errs <- rep(NA_real_, length(ds_eval$sample))
  for (i in seq_along(ds_eval$sample)) {
    df <- as.data.frame(ds_eval$sample[[i]])
    true_params <- ds_eval$target[[i]]
    
    fit <- tryCatch(lm(y ~ ., data = df), error = function(e) NULL)
    if (!is.null(fit) && length(coef(fit)) == ncol(df)) {
      est_beta <- coef(fit)[-1]
      est_c    <- coef(fit)[1]
      est_params <- c(est_beta, est_c)
      errs[i] <- mean((est_params - true_params)^2)
    }
  }
  errs
}

# ==============================================================================
# 3. Regime / Stress Test Loop (NO PAD)
# ==============================================================================
results_hard <- tibble()

for (x_conf in x_configs_simple) {
  for (n_conf in noise_configs) {
    
    cat(sprintf("\nTesting Noise: [%s] ... ", n_conf$name))
    
    # ds_eval must contain:
    # - ds_eval$sample : list of matrices (Li x (d+1)) with columns x1..xd,y  => here d=5 => 6 cols
    # - ds_eval$target : list of numeric vectors length (d+1)                => 6
    ds_eval <- sample_regression_batch(
      n_tasks      = 200,
      L_range      = c(40, 60),
      d            = 5,
      x_dist       = x_conf$dist,
      x_params     = x_conf$params,
      noise_dist   = n_conf$dist,
      noise_params = n_conf$params
    )
    
    # Prepare inputs/targets
    X_test <- to_list_of_tensors(ds_eval$sample, device)  # list[Tensor(Li x 6)]
    Y_true <- do.call(rbind, ds_eval$target)              # (N x 6) numeric matrix
    
    # DeepSets
    with(torch$no_grad(), {
      preds_ds <- model_ds(X_test)$cpu()$numpy()
    })
    mse_ds <- mean((preds_ds - Y_true)^2)
    
    # Transformer
    with(torch$no_grad(), {
      preds_tf <- model_tf(X_test)$cpu()$numpy()
    })
    mse_tf <- mean((preds_tf - Y_true)^2)
    
    # OLS baseline
    ols_err <- ols_errors(ds_eval)
    mse_ols <- if (n_conf$name %in% c("Cauchy", "Pareto")) {
      median(ols_err, na.rm = TRUE)
    } else {
      mean(ols_err, na.rm = TRUE)
    }
    
    cat(sprintf("DS: %.4f | TF: %.4f | OLS: %.4f", mse_ds, mse_tf, mse_ols))
    
    results_hard <- bind_rows(
      results_hard,
      tibble(
        Noise_Type      = n_conf$name,
        MSE_DeepSets    = mse_ds,
        MSE_Transformer = mse_tf,
        MSE_OLS         = mse_ols
      )
    )
  }
}

cat("\n\n=== Final Results Table ===\n")
print(results_hard)
