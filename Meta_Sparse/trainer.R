library(reticulate)

source("Meta_Sparse/data_generator.R")
source("Meta_Sparse/model.R")

device <- if (torch$cuda$is_available()) "cuda" else "cpu"

prepare_batch <- function(batch) {
  list(
    x = lapply(batch, \(d) torch$tensor(cbind(d$X, d$y), dtype = torch$float32)),
    X = lapply(batch, \(d) torch$tensor(d$X, dtype = torch$float32)$to(device)),
    y = lapply(batch, \(d) torch$tensor(d$y, dtype = torch$float32)$to(device))
  )
}

train_epoch <- function(model, opt, data, bs) {
  model$train()
  idx <- sample.int(length(data))
  nb <- floor(length(data) / bs)
  if (nb == 0) return(0)
  loss_sum <- 0
  
  for (b in 1:nb) {
    id <- idx[((b - 1) * bs + 1):(b * bs)]
    batch <- prepare_batch(data[id])
    opt$zero_grad()
    
    out <- model(batch$x)
    mags <- out[[1]]$unbind(0L)
    probs <- out[[2]]$unbind(0L)
    
    loss <- torch$tensor(0, device = device)
    for (i in seq_along(batch$X)) {
      beta <- mags[[i]] * probs[[i]]
      yhat <- torch$matmul(batch$X[[i]], beta)
      loss <- loss + nn$functional$mse_loss(yhat$view_as(batch$y[[i]]), batch$y[[i]])
    }
    
    loss <- loss / length(batch$X)
    loss$backward()
    opt$step()
    loss_sum <- loss_sum + loss$item()
  }
  
  loss_sum / nb
}

eval_epoch <- function(model, data, bs) {
  model$eval()
  nb <- floor(length(data) / bs)
  if (nb == 0) return(0)
  loss_sum <- 0
  
  with(torch$no_grad(), {
    for (b in 1:nb) {
      id <- ((b - 1) * bs + 1):(b * bs)
      batch <- prepare_batch(data[id])
      
      out <- model(batch$x)
      mags <- out[[1]]$unbind(0L)
      probs <- out[[2]]$unbind(0L)
      
      l <- 0
      for (i in seq_along(batch$X)) {
        mask <- (probs[[i]] > 0.5)$float()
        beta <- mags[[i]] * mask
        yhat <- torch$matmul(batch$X[[i]], beta)
        l <- l + nn$functional$mse_loss(
          yhat$view_as(batch$y[[i]]), batch$y[[i]]
        )$item()
      }
      loss_sum <- loss_sum + l / length(batch$X)
    }
  })
  
  loss_sum / nb
}

run_training <- function(epochs = 300, bs = 64, lr = 1e-4) {
  ds <- make_dataset()
  sp <- split_dataset(ds$data_list, ds$indices)
  
  model <- py$TransformerNet(in_dim = ds$p + 1, out_dim = ds$p)$to(device)
  opt <- torch$optim$Adam(model$parameters(), lr = lr)
  
  if (!dir.exists("checkpoints")) dir.create("checkpoints")
  
  for (e in 1:epochs) {
    tr <- train_epoch(model, opt, sp$train_data, bs)
    va <- eval_epoch(model, sp$val_data, bs)
    
    cat(sprintf("Epoch %d | Train %.4f | Val %.4f\n", e, tr, va))
    
    if (e %% 10 == 0) {
      torch$save(model$state_dict(), sprintf("checkpoints/model_%d.pt", e))
    }
  }
}

run_training()
