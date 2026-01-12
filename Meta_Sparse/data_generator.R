generate_sim <- function(k, L_range, p = 100, sigma = 1.0) {
  N <- sample(L_range, 1)
  beta <- numeric(p)
  beta[sample.int(p, k)] <- rnorm(k, 0, 2)
  X <- matrix(rnorm(N * p), nrow = N, ncol = p)
  y <- as.vector(X %*% beta + rnorm(N, 0, sigma))
  list(X = X, y = y, beta = beta, k = k, N = N)
}

make_dataset <- function(p = 100,
                         L_range = 400:500,
                         sparsity_levels = seq(5, 100, 5),
                         reps_per_k = 300,
                         seed = 123) {
  set.seed(seed)
  k_values <- rep(sparsity_levels, each = reps_per_k)
  data_list <- lapply(k_values, function(k) generate_sim(k, L_range, p))
  idx <- sample.int(length(data_list))
  list(data_list = data_list, indices = idx, p = p)
}

split_dataset <- function(data_list, indices, n_train = 5400) {
  n_train <- min(n_train, length(data_list))
  list(
    train_data = data_list[indices[1:n_train]],
    val_data   = data_list[indices[(n_train + 1):length(indices)]]
  )
}
