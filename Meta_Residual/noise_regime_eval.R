# regime_stress_test.R

suppressPackageStartupMessages({
  library(reticulate)
  library(tibble)
  library(dplyr)
})

source("models.R")           # defines py$TransformerNet, py$DeepSetsNet
source("data_generation.R")  # must define sample_regression_batch()
torch <- import("torch")

# --- Device ---
device <- if (torch$cuda$is_available()) torch$device("cuda") else torch$device("cpu")
cat(sprintf("\n--- Starting Stress Test on Device: %s ---\n", device$type))

# --- Instantiate + load weights ---
model_tf <- py$TransformerNet(in_dim=6L, out_dim=6L)$to(device)
model_ds <- py$DeepSetsNet(in_dim=6L, out_dim=6L)$to(device)

cat("Reloading weights...\n")
model_tf$load_state_dict(torch$load("results_r_fixed/tf_n400.pth", map_location=device))
model_ds$load_state_dict(torch$load("results_r_fixed/ds_n500.pth", map_location=device))
cat("Weights loaded.\n")

model_tf$eval()
model_ds$eval()

# ==============================================================================
# 1. Simulation Settings (1 X-dist, 11 Noise Types)
# ==============================================================================
x_configs_simple <- list(
  list(name="Norm_Std", dist="normal", params=list(mean=0, sd=1))
)

noise_configs <- list(
  list(name="Clean",      dist="normal",      params=list(sd=0.1)),
  list(name="Noisy",      dist="normal",      params=list(sd=4.0)),
  list(name="Laplace",    dist="laplace",     params=list(b=1.0)),
  list(name="Pareto",     dist="pareto",      params=list(shape=3.0)),
  list(name="Student_T",  dist="student_t",   params=list(df=3)),
  list(name="Cauchy",     dist="cauchy",      params=list(location=0, scale=1)),
  list(name="Exp_Skew",   dist="exponential", params=list(rate=1.0)),
  list(name="BiModal",    dist="bimodal",     params=list(mu=3, sd=1)),
  list(name="Poisson",    dist="poisson",     params=list(lambda=3)),
  list(name="Pos_Only",   dist="positive",    params=list(rate=1.0)),
  list(name="TriModal",   dist="trimodal",    params=list(mu=4, sd_main=1, sd_side=0.1, p_side=0.1))
)

# ==============================================================================
# 2. Helpers
# ==============================================================================
pad_to_array <- function(sample_list) {
  mats <- lapply(sample_list, function(x) as.matrix(x))
  max_len <- max(sapply(mats, nrow))
  d_in <- ncol(mats[[1]])
  
  X_array <- array(0, dim = c(length(mats), max_len, d_in))
  for (i in seq_along(mats)) {
    m <- mats[[i]]
    X_array[i, 1:nrow(m), ] <- m
  }
  X_array
}

ols_errors <- function(ds_eval) {
  errs <- rep(NA_real_, length(ds_eval$sample))
  for (i in seq_along(ds_eval$sample)) {
    df <- as.data.frame(ds_eval$sample[[i]])
    true_params <- ds_eval$target[[i]]
    
    fit <- tryCatch(lm(y ~ ., data=df), error=function(e) NULL)
    if (!is.null(fit) && length(coef(fit)) == ncol(df)) {
      est_beta <- coef(fit)[-1]
      est_c <- coef(fit)[1]
      est_params <- c(est_beta, est_c)
      errs[i] <- mean((est_params - true_params)^2)
    }
  }
  errs
}

# ==============================================================================
# 3. Regime / Stress Test Loop
# ==============================================================================
results_hard <- tibble()

for (x_conf in x_configs_simple) {
  for (n_conf in noise_configs) {
    
    cat(sprintf("\nTesting Noise: [%s] ... ", n_conf$name))
    
    ds_eval <- sample_regression_batch(
      n_tasks = 200,
      L_range = c(40, 60),
      d = 5,
      x_dist = x_conf$dist,
      x_params = x_conf$params,
      noise_dist = n_conf$dist,
      noise_params = n_conf$params
    )
    
    X_array <- pad_to_array(ds_eval$sample)
    X_test_ts <- torch$tensor(X_array, dtype=torch$float32)$to(device)
    Y_true <- do.call(rbind, ds_eval$target)
    
    # DeepSets
    with(torch$no_grad(), { preds_ds <- model_ds(X_test_ts)$cpu()$numpy() })
    mse_ds <- mean((preds_ds - Y_true)^2)
    
    # Transformer
    with(torch$no_grad(), { preds_tf <- model_tf(X_test_ts)$cpu()$numpy() })
    mse_tf <- mean((preds_tf - Y_true)^2)
    
    # OLS
    ols_err <- ols_errors(ds_eval)
    mse_ols <- if (n_conf$name %in% c("Cauchy", "Pareto")) median(ols_err, na.rm=TRUE) else mean(ols_err, na.rm=TRUE)
    
    cat(sprintf("DS: %.4f | TF: %.4f | OLS: %.4f", mse_ds, mse_tf, mse_ols))
    
    results_hard <- bind_rows(
      results_hard,
      tibble(Noise_Type=n_conf$name, MSE_DeepSets=mse_ds, MSE_Transformer=mse_tf, MSE_OLS=mse_ols)
    )
  }
}

cat("\n\n=== Final Results Table ===\n")
print(results_hard)
