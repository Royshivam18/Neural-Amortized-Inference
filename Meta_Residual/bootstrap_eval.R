# bootstrap_robustness.R

suppressPackageStartupMessages({
  library(reticulate)
  library(tibble)
  library(dplyr)
  library(ggplot2)
})

source("Meta_Residual/model.R")
torch <- import("torch")

device <- if (torch$cuda$is_available()) torch$device("cuda") else torch$device("cpu")
cat("Using device:", device$type, "\n")

# --- Instantiate + load weights ---
model_tf <- py$TransformerNet(in_dim=6L, out_dim=6L)$to(device)
model_ds <- py$DeepSetsNet(in_dim=6L, out_dim=6L)$to(device)

cat("Reloading weights...\n")
model_tf$load_state_dict(torch$load("Meta_Residual/model_res_weights/tf_n400.pth", map_location = device))
model_ds$load_state_dict(torch$load("Meta_Residual/model_res_weights/ds_n400.pth", map_location = device))
cat("Weights loaded.\n")

model_tf$eval()
model_ds$eval()

# ==============================================================================
# 1. Noise + data generation for bootstrap
# ==============================================================================
rbimodal <- function(n, mu=3, sd=1, probs=c(0.5,0.5)) {
  probs <- probs / sum(probs)
  comp <- sample.int(2, n, replace=TRUE, prob=probs)
  rnorm(n, c(-mu, mu)[comp], sd)
}

rtrimodal <- function(n, mu=4, sd_main=1, sd_side=0.1, p_side=0.1) {
  p_c <- 1 - (2 * p_side)
  comp <- sample.int(3, n, replace=TRUE, prob=c(p_c, p_side, p_side))
  means <- c(0, -mu, mu); sds <- c(sd_main, sd_side, sd_side)
  rnorm(n, means[comp], sds[comp])
}

draw_eps <- function(n, conf) {
  np <- conf$params
  switch(conf$dist,
         "normal"      = rnorm(n, 0, np$sd),
         "exponential" = rexp(n, np$rate) - (1/np$rate),
         "bimodal"     = rbimodal(n, np$mu, np$sd, np$probs),
         "trimodal"    = rtrimodal(n, np$mu, np$sd_main, np$sd_side, np$p_side)
  )
}

generate_data_noise_specific <- function(n, beta, c_val, conf) {
  p <- length(beta)
  X <- matrix(rnorm(n * p, 0, 1), nrow=n, ncol=p)
  eps <- draw_eps(n, conf)
  y <- as.numeric(X %*% beta) + c_val + eps
  list(X=X, y=y)
}

noise_configs <- list(
  list(name="Clean_Small", dist="normal",      params=list(sd=0.1)),
  list(name="Clean_High",  dist="normal",      params=list(sd=2)),
  list(name="Right_Skew",  dist="exponential", params=list(rate=1.0)),
  list(name="Bi_Unbal",    dist="bimodal",     params=list(mu=3, sd=1, probs=c(0.8, 0.2))),
  list(name="TriModal",    dist="trimodal",    params=list(mu=4, sd_main=1, sd_side=0.1, p_side=0.1))
)

# ==============================================================================
# 2. Bootstrap loop
# ==============================================================================
p <- 5
sample_sizes <- c(50, 100, 200, 500, 1000)
n_bootstrap_reps <- 30

results <- tibble()

cat("Starting Bootstrap Robustness (Deviation + MSE)...\n")

for (conf in noise_configs) {
  cat(sprintf("\n--- Noise: %s ---\n", conf$name))
  
  set.seed(123)
  true_beta <- rnorm(p, 0, 2)
  true_c <- rnorm(1, 0, 2)
  
  for (n in sample_sizes) {
    cat(sprintf("   n=%d ... ", n))
    
    preds_tf <- matrix(NA_real_, nrow=n_bootstrap_reps, ncol=p)
    preds_ds <- matrix(NA_real_, nrow=n_bootstrap_reps, ncol=p)
    
    for (i in seq_len(n_bootstrap_reps)) {
      sim <- generate_data_noise_specific(n, true_beta, true_c, conf)
      
      input_mat <- cbind(sim$X, sim$y)  # [n,6]
      x_ts <- torch$tensor(input_mat, dtype=torch$float32)$to(device)
      
      with(torch$no_grad(), {
        out_tf <- model_tf(list(x_ts))
        out_ds <- model_ds(list(x_ts))
      })
      
      v_tf <- as.numeric(out_tf$detach()$cpu()$numpy())
      v_ds <- as.numeric(out_ds$detach()$cpu()$numpy())
      
      preds_tf[i,] <- v_tf[1:p]
      preds_ds[i,] <- v_ds[1:p]
    }
    
    dev_tf <- mean(apply(preds_tf, 2, sd))
    dev_ds <- mean(apply(preds_ds, 2, sd))
    mse_tf <- mean((sweep(preds_tf, 2, true_beta, "-"))^2)
    mse_ds <- mean((sweep(preds_ds, 2, true_beta, "-"))^2)
    
    cat(sprintf("TF[Dev %.3f MSE %.3f] | DS[Dev %.3f MSE %.3f]\n", dev_tf, mse_tf, dev_ds, mse_ds))
    
    results <- bind_rows(
      results,
      tibble(Model="Transformer", SampleSize=n, NoiseType=conf$name, Metric="Deviation", Value=dev_tf),
      tibble(Model="Transformer", SampleSize=n, NoiseType=conf$name, Metric="MSE",       Value=mse_tf),
      tibble(Model="DeepSet",     SampleSize=n, NoiseType=conf$name, Metric="Deviation", Value=dev_ds),
      tibble(Model="DeepSet",     SampleSize=n, NoiseType=conf$name, Metric="MSE",       Value=mse_ds)
    )
  }
}

# ==============================================================================
# 3. Two plots (printed only)
# ==============================================================================
results <- results %>%
  mutate(
    NoiseCategory = case_when(
      NoiseType %in% c("Clean_Small", "Clean_High") ~ "Gaussian",
      NoiseType %in% c("Right_Skew") ~ "Skewed",
      NoiseType %in% c("Bi_Unbal", "TriModal") ~ "Multimodal",
      TRUE ~ "Other"
    )
  )

plot_dev <- results %>%
  filter(Metric=="Deviation") %>%
  ggplot(aes(x=SampleSize, y=Value, color=NoiseCategory, linetype=NoiseCategory, shape=NoiseCategory)) +
  stat_summary(fun=mean, geom="line", linewidth=1.1) +
  stat_summary(fun=mean, geom="point", size=3.5) +
  facet_wrap(~Model, scales="free_y") +
  scale_x_log10(breaks=sample_sizes) +
  theme_bw(base_size=14) +
  labs(x="Sample Size (N)", y="Deviation (SD of predicted β)", color=NULL, linetype=NULL, shape=NULL)

plot_mse <- results %>%
  filter(Metric=="MSE") %>%
  ggplot(aes(x=SampleSize, y=Value, color=NoiseCategory, linetype=NoiseCategory, shape=NoiseCategory)) +
  stat_summary(fun=mean, geom="line", linewidth=1.1) +
  stat_summary(fun=mean, geom="point", size=3.5) +
  facet_wrap(~Model, scales="free_y") +
  scale_x_log10(breaks=sample_sizes) +
  theme_bw(base_size=14) +
  labs(x="Sample Size (N)", y="MSE of predicted β", color=NULL, linetype=NULL, shape=NULL)

print(plot_dev)
print(plot_mse)

cat("\nDone.\n")
