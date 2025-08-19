#' Miss.ZIPLNPCA_VE
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



Miss.ZIPLNPCA_VE <- function(Y, # Matrice de comptage 
                              X, # Covariables
                              q, # Dimension de l'espace latent
                              params = NULL, # Paramètres fourni en entrée
                              config_vem = NULL, # (maxiter, tolS, tolxi, ftol, xtol) 
                              config = NULL){ # Configuration pour les étapes
  ## Dimensions
  n <- nrow(Y)
  p <- ncol(Y)
  d <- ncol(X)
  
  ## Configuration
  
  if(is.null(config_vem)){
    config_vem <- list(maxiter = 10000, ftol = 1e-08, xtol = 1e-04, 
                       tolS = list(lower = 1e-04, upper = 1), tolxi = 1e-04)
  }
  
  if (is.null(config)){
    config <- list(algorithm = "MMA", backend = "nlopt", maxeval = 1000,
                   ftol_abs = 1e-8, xtol_abs = 1e-4, maxtime = -1, trace = 1, ftol_rel = 1e-15, xtol_rel = 1e-15)
  }
  
  
  ftol <- config_vem$ftol
  xtol <- config_vem$xtol
  tolS <- config_vem$tolS
  tolxi <- config_vem$tolxi
  maxiter <- config_vem$maxiter
  
  ## Preparation 
  
  R <- ifelse(is.na(Y), 0, 1) # Masque qui met des 0 à la place des données manquantes
  
  Y.na <- ifelse(R == 0, 0, Y)
  
  data <- list(Y = Y.na,
               R = R,
               X = X)
  
  uBound <- c(rep(Inf, (2*d)+(p*q)+(n*q)), rep(tolS$upper, n*q))
  lBound <- c(rep(-Inf, (2*d)+(p*q)+(n*q)), rep(tolS$lower, n*q))
  config$lower_bounds <- lBound
  config$upper_bounds <- uBound
  
  ## Initialisation
  
  if(is.null(params)){
    params <- Init_ZIP(Y, X, q)
  }
  
  outVE <- nlopt_optimize_ZIP_VE(data, params, config, tolxi)
  
  ## Résultats
  out <- outVE
  
  mu <- VectorToMatrix(X%*%out$B, n, p)
  nu <- VectorToMatrix(X%*%out$D, n, p)
  
  mStep <- list(gamma = out$D, beta = out$B, C = out$C)
  eStep <- list(M = out$M, S = out$S,  xi = out$xi)
  
  B.hat <- mStep$beta
  D.hat <- mStep$gamma
  C.hat <- mStep$C
  M.hat <- eStep$M
  S.hat <- eStep$S
  XB.hat <- VectorToMatrix(X %*% B.hat, n, p)
  XD.hat <- VectorToMatrix(X %*% D.hat, n, p)
  
  predicted <- exp(XB.hat + M.hat %*% t(C.hat) + 0.5 * (S.hat) %*% t(C.hat * C.hat))
  elbo1 <- out$elbo1 ; elbo2 <- out$elbo2 ; elbo3 <- out$elbo3
  elbo4 <- out$elbo4 ; elbo5 <- out$elbo5
  
  pred <- list(A = out$A, nu = nu, mu = mu, predicted = predicted)
  # elbo <- out$objective_values[length(out$objective_values)]
  elbo <- out$objective
  
  
  res <- list(mStep = mStep,
              eStep = eStep,
              pred = pred,
              elbo = elbo,
              params.init = params,
              monitoring = out$monitoring,
              elbo1 = elbo1,
              elbo2 = elbo2,
              elbo3 = elbo3,
              elbo4 = elbo4,
              elbo5 = elbo5)
}