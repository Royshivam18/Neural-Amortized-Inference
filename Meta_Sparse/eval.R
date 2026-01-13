library(reticulate)
library(ggplot2)

source("Meta_Sparse/model.R")
torch <- import("torch")

device <- if (torch$cuda$is_available()) "cuda" else "cpu"
p <- 100L

model <- py$TransformerNet(in_dim = as.integer(p + 1L),
                           out_dim = as.integer(p))$to(device)

state_dict <- torch$load("Meta_Sparse/checkpoints/model_epoch_200.pt",
                         map_location = device)

model$load_state_dict(state_dict)
model$eval()

sample_sizes <- c(50, 100, 200, 500, 1000)
sparsity_levels <- c(5, 20, 50, 80)
n_bootstrap_reps <- 30
p <- 100

generate_sparse_beta <- function(p, k) {
  beta <- rep(0, p)
  if (k > 0) {
    idx <- sample.int(p, k)
    vals <- runif(k, -2, 2)
    small <- abs(vals) < 0.5
    vals[small] <- 0.5 * sign(vals[small])
    beta[idx] <- vals
  }
  beta
}

generate_data_robust <- function(n, beta, sigma = 1.0) {
  X <- matrix(rnorm(n * length(beta)), nrow = n)
  y <- as.numeric(X %*% beta + rnorm(n, 0, sigma))
  list(X = X, y = y)
}

eval_once <- function(model, X, y, device, prob_thresh = 0.5) {
  input_tensor <- torch$tensor(cbind(X, y), dtype = torch$float32, device = device)
  out <- model(list(input_tensor))
  pred_mag <- out[[1]]
  pred_prob <- out[[2]]
  mask <- (pred_prob > prob_thresh)$float()
  final_beta <- pred_mag * mask
  as.numeric(final_beta$cpu()$numpy())
}

model$eval()

results <- data.frame()

cat("Running Experiment for Sparsities: 5, 20, 50, 80...\n")

for (k in sparsity_levels) {
  cat(sprintf("\n--- Sparsity k = %d ---\n", k))
  
  true_beta <- generate_sparse_beta(p, k)
  
  for (n in sample_sizes) {
    cat(sprintf("   n = %d ... ", n))
    
    preds <- matrix(0, nrow = n_bootstrap_reps, ncol = p)
    
    with(torch$no_grad(), {
      for (i in 1:n_bootstrap_reps) {
        sim <- generate_data_robust(n, true_beta)
        preds[i, ] <- eval_once(model, sim$X, sim$y, device)
      }
    })
    
    avg_dev <- mean(apply(preds, 2, sd))
    cat(sprintf("Avg Dev: %.4f\n", avg_dev))
    
    results <- rbind(
      results,
      data.frame(
        SampleSize = n,
        AvgDeviation = avg_dev,
        Sparsity = as.factor(k)
      )
    )
  }
}

plt <- ggplot(results, aes(x = SampleSize, y = AvgDeviation, color = Sparsity, group = Sparsity)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  scale_color_viridis_d(option = "plasma", end = 0.9, name = "Sparsity (k)") +
  labs(
    title = "Bootstrap Variance of Beta vs Sample Size",
    subtitle = "Comparing Estimator Stability across different Sparsity Levels",
    x = "Sample Size (n)",
    y = "Avg Bootstrap SD of Beta"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")


results
