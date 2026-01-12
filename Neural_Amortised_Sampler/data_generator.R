# ==============================================================================
# data_generator.R
# ==============================================================================

# Generates ONE task from the 8-mode "ring of Gaussians" prior and returns:
# - params:       (1 x 2) matrix  [beta_1, beta_2]
# - context_flat: (1 x 2*n_points) matrix [x1..xN, y1..yN]
generate_8mode_task <- function(
    n_points   = 20,
    R          = 5.0,
    sigma_mode = 0.5,
    sigma_eps  = 0.5,
    x_min      = -2,
    x_max      =  2
) {
  mode_idx <- sample(0:7, 1)
  angle <- (2 * pi * mode_idx) / 8
  
  mu_beta1 <- R * cos(angle)
  mu_beta2 <- R * sin(angle)
  
  beta_1 <- rnorm(1, mean = mu_beta1, sd = sigma_mode)
  beta_2 <- rnorm(1, mean = mu_beta2, sd = sigma_mode)
  
  x <- runif(n_points, min = x_min, max = x_max)
  eps <- rnorm(n_points, mean = 0, sd = sigma_eps)
  y <- beta_1 * x + beta_2 + eps
  
  list(
    params = matrix(c(beta_1, beta_2), nrow = 1),
    context_flat = matrix(c(x, y), nrow = 1)
  )
}

# Builds a full dataset of tasks:
# Returns list(contexts, params)
# - contexts: (N_tasks x 2*n_points) matrix
# - params:   (N_tasks x 2) matrix
generate_dataset <- function(
    n_tasks    = 5000,
    n_points   = 20,
    R          = 5.0,
    sigma_mode = 0.5,
    sigma_eps  = 0.5,
    x_min      = -2,
    x_max      =  2,
    seed       = NULL
) {
  if (!is.null(seed)) set.seed(seed)
  
  tasks <- lapply(seq_len(n_tasks), function(i) {
    generate_8mode_task(
      n_points   = n_points,
      R          = R,
      sigma_mode = sigma_mode,
      sigma_eps  = sigma_eps,
      x_min      = x_min,
      x_max      = x_max
    )
  })
  
  contexts <- do.call(rbind, lapply(tasks, function(t) t$context_flat))
  params   <- do.call(rbind, lapply(tasks, function(t) t$params))
  
  list(contexts = contexts, params = params)
}
