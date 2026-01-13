# ==============================================================================
# FILE 2: generative.R  (task generator for latent centroid mixture)
# ==============================================================================
library(reticulate)


init_centroids <- function(K, p=20, tau=3.0, seed=1L, device=NULL) {
  if (!is.null(seed)) torch$manual_seed(as.integer(seed))
  if (is.null(device)) device <- if (torch$cuda$is_available()) "cuda" else "cpu"
  mu <- torch$randn(c(K, p), dtype=torch$float32) * tau
  mu$to(device)
}

sample_task <- function(centroids, sigma=1.0, N_min=10L, N_max=30L) {
  K <- as.integer(centroids$shape[[1]])
  p <- as.integer(centroids$shape[[2]])
  
  k_idx <- as.integer(sample.int(K, 1))
  beta  <- centroids[k_idx - 1L, ]  # 0-indexed on python side
  
  N_t <- as.integer(sample(N_min:N_max, 1))
  x <- torch$randn(c(N_t, p), dtype=torch$float32, device=beta$device)
  eps <- torch$randn(c(N_t), dtype=torch$float32, device=beta$device) * sigma
  y <- torch$matmul(x, beta$unsqueeze(1))$squeeze(1) + eps
  
  xy <- torch$cat(list(x, y$unsqueeze(1)), dim=1)  # [N_t, p+1]
  
  list(
    beta = beta,     # [p]
    x    = x,        # [N_t, p]
    y    = y,        # [N_t]
    xy   = xy,       # [N_t, p+1]  (the set input)
    k    = k_idx,    # 1..K (R index)
    N    = N_t
  )
}

sample_task_batch <- function(B, centroids, sigma=1.0, N_min=10L, N_max=30L) {
  tasks <- vector("list", B)
  x_list <- vector("list", B)
  betas  <- vector("list", B)
  
  for (b in seq_len(B)) {
    tb <- sample_task(centroids, sigma=sigma, N_min=N_min, N_max=N_max)
    tasks[[b]] <- tb
    x_list[[b]] <- tb$xy
    betas[[b]]  <- tb$beta$unsqueeze(0)  # [1,p]
  }
  
  y_beta <- torch$cat(betas, dim=0)      # [B,p]
  list(x_list=x_list, y_beta=y_beta, tasks=tasks)
}
