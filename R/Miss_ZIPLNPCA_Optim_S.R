#' Miss.ZIPLNPCA_Optim.S
#'
#' Estimation of the parameters and the missing data
#' @param Y count matrix
#' @param X covariates
#' @param q size of the latent space
#' @param params Initial parameters
#' @param config configuration of the optimizer
#' @param tolS S tolerance
#' @return A list of the estimated parameters
#' @import PLNmodels
#' @export



Miss.ZIPLNPCA_Optim.S <- function(Y, # Table de comptages n*p qui peut contenir des données manquantes
                          X, # Covariables np*d dont une colonne de 1 pour l'intercept
                          q, # Dimension de l'espace latent q
                          params = NULL, # Paramètres fourni en entrée
                          config = NULL,
                          tolS = NULL,
                          tolxi = NULL){
  
  n <- nrow(Y)
  p <- ncol(Y)
  d <- ncol(X)
  
  #if (is.null(config)){config <- PLNPCA_param()$config_optim}
  if (is.null(config)){
    config <- list(algorithm = "MMA", backend = "nlopt", maxeval = 10000,
                   ftol_abs = 1e-8, xtol_abs = 1e-4, maxtime = -1, trace = 1, ftol_rel = 1e-15, xtol_rel = 1e-15)
  }
  if (is.null(tolS)){tolS <- list(lower = 1e-04, upper = 1)}
  if (is.null(tolxi)){tolxi <- 1e-04}
  
  R <- ifelse(is.na(Y), 0, 1) # Masque qui met des 0 à la place des données manquantes
  
  Y.na <- ifelse(R == 0, 0, Y)
  
  data <- list(Y = Y.na,
               R = R,
               X = X)
  
  uBound <- c(rep(Inf, (2*d)+(p*q)+(n*q)), rep(tolS$upper, n*q))
  lBound <- c(rep(-Inf, (2*d)+(p*q)+(n*q)), rep(tolS$lower, n*q))
  config$lower_bounds <- lBound
  config$upper_bounds <- uBound

  
    if (is.null(params)){params <- Init_ZIP(Y, X, q)}
    
    out <- nlopt_optimize_S(data, params, config, tolxi)
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
    
    predicted <- exp(XB.hat + M.hat %*% t(C.hat) + 0.5 * (S.hat*S.hat) %*% t(C.hat * C.hat))
    elbo1 <- out$elbo1 ; elbo2 <- out$elbo2 ; elbo3 <- out$elbo3
    elbo4 <- out$elbo4 ; elbo5 <- out$elbo5

  
  pred <- list(A = out$A, nu = nu, mu = mu, predicted = predicted)
  iter <- out$monitoring$iterations
  elboPath <- out$objective_values
  # elbo <- out$objective_values[length(out$objective_values)]
  elbo <- out$objective

  res <- list(mStep = mStep,
              eStep = eStep,
              pred = pred,
              iter = iter,
              elboPath = elboPath,
              elbo = elbo,
              params.init = params,
              monitoring = out$monitoring,
              elbo1 = elbo1,
              elbo2 = elbo2,
              elbo3 = elbo3,
              elbo4 = elbo4,
              elbo5 = elbo5)
  
  return(res)
  
}
