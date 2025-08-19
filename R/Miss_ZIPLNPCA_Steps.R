#' Miss.ZIPLNPCA_Steps
#'
#' Estimation of the parameters and the missing data
#' @param Y count matrix
#' @param X covariates
#' @param q size of the latent space
#' @param params Initial parameters
#' @param config_vem configuration of the optimizer
#' @param config configuration for the steps 
#' @return A list of the estimated parameters
#' @import PLNmodels
#' @export



Miss.ZIPLNPCA_Steps <- function(Y, X, q, params = NULL, config_vem = NULL, config = NULL) {
  n <- nrow(Y)
  p <- ncol(Y)
  d <- ncol(X)
 
  
  if (is.null(config_vem)) {
    config_vem <- list(maxiter = 1e06, ftol = 1e-10, xtol = 1e-10,
                       tolS = list(lower = 0, upper = Inf), tolxi = 1e-04)
  }
  
  if (is.null(config)) {
    config <- list(algorithm = "MMA", backend = "nlopt", maxeval = 1e04,
                   ftol_abs = 1e-10, xtol_abs = 1e-10, maxtime = -1, trace = 1, ftol_rel = 1e-15, xtol_rel = 1e-15)
  }
  
  R <- ifelse(is.na(Y), 0, 1)
  Y.na <- ifelse(R == 0, 0, Y)
  
  data <- list(Y = Y.na, R = R, X = X)
  
  uBound <- c(rep(Inf, (2*d)+(p*q)+(n*q)), rep(config_vem$tolS$upper, n*q))
  lBound <- c(rep(-Inf, (2*d)+(p*q)+(n*q)), rep(config_vem$tolS$lower, n*q))
  config$upper_bounds <- uBound
  config$lower_bounds <- lBound
  
  if (is.null(params)) {
    params <- Init_ZIP(Y, X, q)
  }
  
  params.init <- params
  params_new <- unlist(params)
  params_old <- params_new + 1
  
  elbo_new <- Elbo_grad(data, params, config_vem$tolxi)$objective
  elbo_old <- elbo_new + 1
  iter <- 0
  elboPath <- c(elbo_new)
  status <- NULL
  
  while ((max(abs(params_new - params_old)) > config_vem$xtol &&
          abs(elbo_new - elbo_old) > config_vem$ftol) &&
         iter < config_vem$maxiter) {
    
    iter <- iter + 1
    cat("Iteration:", iter, "ELBO:", elbo_new, "Diff params:", 
        max(abs(params_new - params_old)), "Diff ELBO:", 
        abs(elbo_new - elbo_old), "\n")
    
    params_old <- params_new
    elbo_old <- elbo_new
    
    for (block in c("B", "D", "C", "M", "S")) {
      
      active_blocks <- list(B = FALSE, D = FALSE, C = FALSE, M = FALSE, S = FALSE)
      active_blocks[[block]] <- TRUE
      
      out_opt <- nlopt_optimize_ZIP_Steps(data, params, config, config_vem$tolxi, active_blocks)
      params$B <- out_opt$B
      params$D <- out_opt$D
      params$C <- out_opt$C
      params$M <- out_opt$M
      params$S <- out_opt$S
      
      grad <- Elbo_grad(data, params, config_vem$tolxi)
      cat(sprintf("gradB : %.4e gradD : %.4e gradC : %.4e gradM : %.4e gradS : %.4e\n",
                  mean(grad$gradB^2), mean(grad$gradD^2),
                  mean(grad$gradC^2), mean(grad$gradM^2),
                  mean(grad$gradS^2)))
    }
    
    params_new <- unlist(params)
    elbo_new <- out_opt$objective
    elboPath <- c(elboPath, elbo_new)
    
    if (max(abs(params_new - params_old)) <= config_vem$xtol) {
      status <- "xtol atteint"
      break
    }
    if (abs(elbo_new - elbo_old) <= config_vem$ftol) {
      status <- "ftol atteint"
      break
    }
  }
  
  if (is.null(status)) status <- "maxiter atteint"
  
  mu <- VectorToMatrix(X %*% out_opt$B, n, p)
  nu <- VectorToMatrix(X %*% out_opt$D, n, p)
  predicted <- exp(mu + out_opt$M %*% t(out_opt$C) + 
                     0.5 * out_opt$S %*% t(out_opt$C^2))
  
  imputed <- ifelse(is.na(Y), out_opt$xi*out_opt$A, Y)
  params <- list(B = out_opt$B, D = out_opt$D, C = out_opt$C, 
                 M = out_opt$M, S = out_opt$S)
  grad <- Elbo_grad(data, params, config_vem$tolxi)
  
  res <- list(
    mStep = list(gamma = out_opt$D, beta = out_opt$B, C = out_opt$C),
    eStep = list(M = out_opt$M, S = out_opt$S, xi = out_opt$xi),
    pred = list(A = out_opt$A, mu = mu, nu = nu, predicted = predicted),
    imputed = imputed,
    iter = iter,
    elboPath = elboPath,
    elbo = out_opt$objective,
    params.init = params.init,
    monitoring = list(status = status, iterations = iter),
    gradB = grad$gradB,
    gradD = grad$gradD,
    gradC = grad$gradC,
    gradM = grad$gradM, 
    gradS = grad$gradS
  )
  
  return(res)
}
