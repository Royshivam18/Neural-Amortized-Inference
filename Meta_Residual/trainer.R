library(reticulate)

use_condaenv("r-amortised", required = TRUE)

torch <- import("torch")
nn    <- import("torch.nn")
# ------------------------------------------------------------------
# LOAD MODELS (IMPORTANT LINE)
# ------------------------------------------------------------------
source("Meta_Residual/model.R")   # makes py$TransformerNet, py$DeepSetsNet available
source("Meta_Residual/data_generator.R")

model_trainer <- function(model, X_train, Y_train,
                          X_val = NULL, Y_val = NULL,
                          epochs = 100L, batch_size = 128L,
                          lr = 1e-3, device = "cuda",
                          save_path = "best_model.pth",
                          verbose_every = 10L) {
  
  model$to(device)
  opt <- torch$optim$Adam(model$parameters(), lr = lr)
  loss_fn <- torch$nn$MSELoss()
  
  Y_train <- Y_train$to(device)
  if (!is.null(Y_val)) Y_val <- Y_val$to(device)
  
  get_n <- function(d) if (inherits(d, "torch.Tensor")) d$size(0L) else length(d)
  n_train <- get_n(X_train)
  has_val <- !is.null(X_val) && !is.null(Y_val)
  n_val   <- if (has_val) get_n(X_val) else 0L
  
  history <- data.frame(epoch=integer(), train_mse=double(), val_mse=double())
  best_val <- Inf
  
  cat(sprintf("Training on %d | Validation on %d\n", n_train, n_val))
  
  for (e in seq_len(epochs)) {
    model$train()
    perm <- sample.int(n_train)
    tr_sum <- 0; tr_steps <- 0
    
    for (s in seq(1L, n_train, by=batch_size)) {
      end_idx <- min(s + batch_size - 1L, n_train)
      idx <- perm[s:end_idx]
      
      xb <- if (inherits(X_train, "list")) {
        X_train[idx]
      } else {
        t_idx <- torch$tensor(idx - 1L, dtype=torch$long)$to(device)
        X_train$index_select(0L, t_idx)
      }
      
      yb <- Y_train$index_select(
        0L, torch$tensor(idx - 1L, dtype=torch$long)$to(device)
      )
      
      opt$zero_grad()
      loss <- loss_fn(model(xb), yb)
      loss$backward()
      opt$step()
      
      tr_sum <- tr_sum + loss$item()
      tr_steps <- tr_steps + 1L
    }
    
    tr_mse <- tr_sum / tr_steps
    val_mse <- NA_real_
    
    if (has_val) {
      model$eval()
      v_sum <- 0; v_steps <- 0
      with(torch$no_grad(), {
        for (s in seq(1L, n_val, by=batch_size)) {
          end_idx <- min(s + batch_size - 1L, n_val)
          idx <- s:end_idx
          
          xb <- X_val[idx]
          yb <- Y_val$index_select(
            0L, torch$tensor(idx - 1L, dtype=torch$long)$to(device)
          )
          
          v_sum <- v_sum + loss_fn(model(xb), yb)$item()
          v_steps <- v_steps + 1L
        }
      })
      
      val_mse <- v_sum / v_steps
      if (val_mse < best_val) {
        best_val <- val_mse
        torch$save(model$state_dict(), save_path)
      }
    }
    
    history <- rbind(history,
                     data.frame(epoch=e, train_mse=tr_mse, val_mse=val_mse))
    
    if (e == 1L || e %% verbose_every == 0L)
      cat(sprintf("Ep %3d | Tr %.5f | Val %.5f\n", e, tr_mse, val_mse))
  }
  
  if (has_val) model$load_state_dict(torch$load(save_path))
  list(model=model, history=history)
}

model_tf <- py$TransformerNet(in_dim=6L, out_dim=6L)
model_ds <- py$DeepSetsNet(in_dim=6L, out_dim=6L)


df <- generate_grid(n_repeats = 200, p = 5, L_range = 10:100, seed = 123)

# Train/val split
set.seed(1)
n <- nrow(df)
id <- sample.int(n)
n_tr <- floor(0.8 * n)

tr_id  <- id[1:n_tr]
val_id <- id[(n_tr + 1):n]

# Raw splits from generator (lists)
X_tr_raw  <- df$sample[tr_id]   # list of matrices (Li x (p+1))
Y_tr_raw  <- df$target[tr_id]   # list of numeric vectors (p+1)

X_val_raw <- df$sample[val_id]
Y_val_raw <- df$target[val_id]

# Device (robust)
device <- if (torch$cuda$is_available()) torch$device("cuda") else torch$device("cpu")

# Convert X: list(matrices) -> list(torch tensors), each (Li x (p+1))
to_list_of_tensors <- function(x_list, device) {
  lapply(x_list, function(M) {
    torch$tensor(M, dtype = torch$float32)$to(device)
  })
}

X_tr  <- to_list_of_tensors(X_tr_raw, device)
X_val <- to_list_of_tensors(X_val_raw, device)

# Convert Y: list(vectors) -> matrix -> torch tensor (N x (p+1))
Y_tr  <- torch$tensor(do.call(rbind, Y_tr_raw),  dtype = torch$float32)$to(device)
Y_val <- torch$tensor(do.call(rbind, Y_val_raw), dtype = torch$float32)$to(device)


res_tf <- model_trainer(model_tf, X_tr, Y_tr, X_val, Y_val, epochs = 10L)
res_ds <- model_trainer(model_ds, X_tr, Y_tr, X_val, Y_val, epochs = 10L)
