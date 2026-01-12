# ==============================================================================
# benchmark.R — FINAL BENCHMARK: Samples & Time Comparison
# ==============================================================================

# ---- deps ----
suppressPackageStartupMessages({
  library(reticulate)
  library(ggplot2)
  library(viridis)
  library(rjags)
  library(coda)
})

# If you use modular files:
# source("data_generator.R")  # must define generate_8mode_task()
# source("model.R")           # must define py$sample_batch_ode() and py$device
# source("trainer.R")         # if you want to train here; otherwise ensure `model` exists

# IMPORTANT: ensure torch imported (R side) for tensor creation
torch <- import("torch")

# Sanity checks
if (!exists("generate_8mode_task")) stop("generate_8mode_task() not found. source('data_generator.R') first.")
if (is.null(py$sample_batch_ode))  stop("py$sample_batch_ode not found. source('model.R') first.")
if (!exists("model"))              stop("`model` not found. Train/load your flow model first (trainer.R).")

cat("\n--- Starting Final Benchmark ---\n")

# ------------------------------------------------------------------------------
# 1. SETUP: Create 200 distinct tasks
# ------------------------------------------------------------------------------
set.seed(123)
n_bench_tasks   <- 200L
points_per_task <- 20L

tasks_list <- lapply(seq_len(n_bench_tasks), function(i) {
  generate_8mode_task(n_points = points_per_task)
})

# Prepare JAGS matrices x[t,i], y[t,i]
x_mat <- matrix(NA_real_, nrow = n_bench_tasks, ncol = points_per_task)
y_mat <- matrix(NA_real_, nrow = n_bench_tasks, ncol = points_per_task)

for (i in seq_len(n_bench_tasks)) {
  raw <- as.numeric(tasks_list[[i]]$context_flat) # 1 x (2N)
  x_mat[i, ] <- raw[1:points_per_task]
  y_mat[i, ] <- raw[(points_per_task + 1L):(2L * points_per_task)]
}

# Prepare Flow input tensor (Batch x 2N)
ctx_flow   <- do.call(rbind, lapply(tasks_list, function(t) t$context_flat))
ctx_tensor <- torch$tensor(ctx_flow, dtype = torch$float32)$to(py$device)

# ------------------------------------------------------------------------------
# 2. RUN MCMC (JAGS)
# ------------------------------------------------------------------------------
cat("Running MCMC (Batch Mode)...\n")

jags_str <- "
model {
  for (t in 1:n_tasks) {
    z[t] ~ dcat(pi_probs)          # mixture component (1..8)
    beta1[t] ~ dnorm(mu1[z[t]], tau_mode)
    beta2[t] ~ dnorm(mu2[z[t]], tau_mode)

    for (i in 1:N_pts) {
      y[t, i] ~ dnorm(beta1[t] * x[t, i] + beta2[t], tau_eps)
    }
  }
}
"

# Prior ring centers
R_rad  <- 5.0
angles <- (0:7) * (2 * pi / 8)

# NOTE: In JAGS, tau = precision = 1/sigma^2
# Your sigma_mode=0.5 => tau_mode = 1/0.25 = 4
# Your sigma_eps=0.5  => tau_eps  = 4
j_data <- list(
  n_tasks  = n_bench_tasks,
  N_pts    = points_per_task,
  x        = x_mat,
  y        = y_mat,
  mu1      = R_rad * cos(angles),
  mu2      = R_rad * sin(angles),
  tau_mode = 4.0,
  tau_eps  = 4.0,
  pi_probs = rep(1/8, 8)
)

# Compile (setup time not counted)
mod <- jags.model(textConnection(jags_str), data = j_data, n.chains = 1, quiet = TRUE)

# Burn-in (not counted)
update(mod, n.iter = 500, progress.bar = "none")

# --- time MCMC sampling ---
start_mcmc <- Sys.time()
samps <- coda.samples(mod, variable.names = c("beta1", "beta2"), n.iter = 1000, progress.bar = "none")
end_mcmc <- Sys.time()
time_mcmc <- as.numeric(difftime(end_mcmc, start_mcmc, units = "secs"))

# Take last 5 iterations per task
mat_samps <- as.matrix(samps[[1]])
idx_last5 <- (nrow(mat_samps) - 4L):nrow(mat_samps)

mcmc_df <- do.call(rbind, lapply(seq_len(n_bench_tasks), function(t) {
  data.frame(
    beta_1 = mat_samps[idx_last5, paste0("beta1[", t, "]")],
    beta_2 = mat_samps[idx_last5, paste0("beta2[", t, "]")],
    stringsAsFactors = FALSE
  )
}))
mcmc_df$Method <- "Metropolis-Hastings (JAGS)"

cat(sprintf("MCMC done. Time: %.2f s\n", time_mcmc))

# ------------------------------------------------------------------------------
# 3. RUN FLOW MATCHING
# ------------------------------------------------------------------------------
cat("Running Flow Matching...\n")

start_flow <- Sys.time()
batch_input <- ctx_tensor$repeat_interleave(5L, dim = 0L) # 5 samples per task
flow_out_np <- py$sample_batch_ode(model, batch_input, steps = 50L)
end_flow <- Sys.time()
time_flow <- as.numeric(difftime(end_flow, start_flow, units = "secs"))

flow_df <- as.data.frame(flow_out_np)
colnames(flow_df) <- c("beta_1", "beta_2")
flow_df$Method <- "Normalizing Flow"

cat(sprintf("Flow done. Time: %.2f s\n", time_flow))

# ------------------------------------------------------------------------------
# 4. PLOT RESULTS
# ------------------------------------------------------------------------------
all_data <- rbind(mcmc_df, flow_df)

# Plot 1: JAGS
jags_data <- subset(all_data, Method == "Metropolis-Hastings (JAGS)")
p_jags <- ggplot(jags_data, aes(x = beta_1, y = beta_2)) +
  stat_density_2d(geom = "raster", aes(fill = after_stat(density)), contour = FALSE, n = 200) +
  geom_point(alpha = 0.2, color = "white", size = 0.2) +
  scale_fill_viridis_c(option = "viridis") +
  coord_fixed() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  labs(
    title = sprintf("Metropolis-Hastings (JAGS) | Time: %.2f s", time_mcmc),
    x = expression(beta[1]),
    y = expression(beta[2]),
    fill = "Density"
  ) +
  theme_minimal(base_size = 16) +
  theme(panel.grid = element_blank(),
        plot.title = element_text(face = "bold"))

# Plot 2: Flow
flow_data <- subset(all_data, Method == "Normalizing Flow")
p_flow <- ggplot(flow_data, aes(x = beta_1, y = beta_2)) +
  stat_density_2d(geom = "raster", aes(fill = after_stat(density)), contour = FALSE, n = 200) +
  geom_point(alpha = 0.2, color = "white", size = 0.2) +
  scale_fill_viridis_c(option = "viridis") +
  coord_fixed() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  labs(
    title = sprintf("Amortised Flow | Time: %.2f s", time_flow),
    x = expression(beta[1]),
    y = expression(beta[2]),
    fill = "Density"
  ) +
  theme_minimal(base_size = 16) +
  theme(panel.grid = element_blank(),
        plot.title = element_text(face = "bold"))

ggsave("plot_jags_final.png", p_jags, width = 8, height = 6)
ggsave("plot_flow_final.png", p_flow, width = 8, height = 6)

print(p_jags)
print(p_flow)

cat("\n--- Benchmark Complete ---\n")
cat(sprintf("JAGS sampling time: %.2f s\n", time_mcmc))
cat(sprintf("Flow sampling time: %.2f s\n", time_flow))
