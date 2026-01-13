library(tibble)


rbimodal <- function(n, mu = 3, sd = 1, probs = c(0.5, 0.5)) {
  stopifnot(n >= 0, length(probs) == 2, all(probs >= 0), sum(probs) > 0)
  probs <- probs / sum(probs)
  comp <- sample.int(2, size = n, replace = TRUE, prob = probs)
  rnorm(n, mean = c(-mu, mu)[comp], sd = sd)
}

rtrimodal <- function(n, mu = 4, sd_main = 1, sd_side = 0.1, p_side = 0.1) {
  stopifnot(n >= 0, p_side >= 0, (2 * p_side) <= 1)
  p_c <- 1 - (2 * p_side)
  comp <- sample.int(3, size = n, replace = TRUE, prob = c(p_c, p_side, p_side))
  means <- c(0, -mu, mu)
  sds   <- c(sd_main, sd_side, sd_side)
  rnorm(n, mean = means[comp], sd = sds[comp])
}

draw_X <- function(L, p, dist, params) {
  stopifnot(L > 0, p > 0)
  if (dist == "normal") {
    mu <- if (!is.null(params$mean)) params$mean else 0
    sd <- if (!is.null(params$sd))   params$sd   else 1
    return(matrix(rnorm(L * p, mean = mu, sd = sd), nrow = L, ncol = p))
  }
  stop(sprintf("Unsupported x distribution: %s", dist))
}

draw_eps <- function(L, dist, params) {
  stopifnot(L > 0)
  if (dist == "normal") {
    sd <- if (!is.null(params$sd)) params$sd else 1
    return(rnorm(L, mean = 0, sd = sd))
  }
  if (dist == "exponential") {
    rate <- if (!is.null(params$rate)) params$rate else 1
    return(rexp(L, rate = rate) - (1 / rate))
  }
  if (dist == "skew_left") {
    rate <- if (!is.null(params$rate)) params$rate else 1
    return(-(rexp(L, rate = rate) - (1 / rate)))
  }
  if (dist == "bimodal") {
    mu    <- if (!is.null(params$mu))    params$mu    else 3
    sd    <- if (!is.null(params$sd))    params$sd    else 1
    probs <- if (!is.null(params$probs)) params$probs else c(0.5, 0.5)
    return(rbimodal(L, mu = mu, sd = sd, probs = probs))
  }
  if (dist == "trimodal") {
    mu       <- if (!is.null(params$mu))       params$mu       else 4
    sd_main  <- if (!is.null(params$sd_main))  params$sd_main  else 1
    sd_side  <- if (!is.null(params$sd_side))  params$sd_side  else 0.1
    p_side   <- if (!is.null(params$p_side))   params$p_side   else 0.1
    return(rtrimodal(L, mu = mu, sd_main = sd_main, sd_side = sd_side, p_side = p_side))
  }
  stop(sprintf("Unsupported noise distribution: %s", dist))
}

make_task <- function(L, p, x_cfg, noise_cfg, beta_sd = 9, c_sd = 9) {
  beta <- rnorm(p, mean = 0, sd = beta_sd)
  cval <- rnorm(1, mean = 0, sd = c_sd)
  
  X <- draw_X(L, p, dist = x_cfg$dist, params = x_cfg$params)
  eps <- draw_eps(L, dist = noise_cfg$dist, params = noise_cfg$params)
  
  y <- as.numeric(X %*% beta) + cval + eps
  
  sample_mat <- cbind(X, y)
  colnames(sample_mat) <- c(paste0("x", seq_len(p)), "y")
  
  list(
    sample = sample_mat,
    target = c(beta, cval)
  )
}

default_x_configs <- function() {
  list(
    list(name = "Norm_Std", dist = "normal", params = list(mean = 0, sd = 1))
  )
}

default_noise_configs <- function() {
  list(
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
}

generate_grid <- function(
    n_repeats,
    p = 5,
    L_range = 10:1000,
    x_configs = default_x_configs(),
    noise_configs = default_noise_configs(),
    seed = NULL,
    beta_sd = 9,
    c_sd = 9
) {
  stopifnot(n_repeats >= 1, p >= 1, length(L_range) >= 1)
  if (!is.null(seed)) set.seed(seed)
  
  tasks <- vector("list", length = n_repeats * length(x_configs) * length(noise_configs))
  idx <- 1L
  
  for (r in seq_len(n_repeats)) {
    for (xc in x_configs) {
      for (nc in noise_configs) {
        L <- sample(L_range, size = 1)
        tasks[[idx]] <- make_task(
          L = L, p = p,
          x_cfg = xc, noise_cfg = nc,
          beta_sd = beta_sd, c_sd = c_sd
        )
        idx <- idx + 1L
      }
    }
  }
  
  tibble(
    sample = lapply(tasks, `[[`, "sample"),
    target = lapply(tasks, `[[`, "target")
  )
}

# Example:
# df <- generate_grid(n_repeats = 3, seed = 123)
# str(df)


# ------------------------------------------------------------------------------
# Helper: sample_regression_batch()  (COMPATIBLE WITH YOUR make_task())
# Returns a list: $sample (list of matrices) and $target (list of vectors)
# ------------------------------------------------------------------------------

sample_regression_batch <- function(
    n_tasks = 200,
    L_range = c(40, 60),
    d = 5,
    x_dist = "normal",
    x_params = list(mean = 0, sd = 1),
    noise_dist = "normal",
    noise_params = list(sd = 0.1),
    beta_sd = 9,
    c_sd = 9,
    seed = NULL
) {
  stopifnot(n_tasks >= 1, length(L_range) == 2, L_range[1] <= L_range[2])
  if (!is.null(seed)) set.seed(seed)
  
  x_cfg    <- list(dist = x_dist,   params = x_params)
  noise_cfg <- list(dist = noise_dist, params = noise_params)
  
  tasks <- vector("list", n_tasks)
  for (i in seq_len(n_tasks)) {
    L_i <- sample(seq.int(L_range[1], L_range[2]), size = 1)
    tasks[[i]] <- make_task(
      L = L_i, p = d,
      x_cfg = x_cfg,
      noise_cfg = noise_cfg,
      beta_sd = beta_sd,
      c_sd = c_sd
    )
  }
  
  list(
    sample = lapply(tasks, `[[`, "sample"),
    target = lapply(tasks, `[[`, "target")
  )
}
