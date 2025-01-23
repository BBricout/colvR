#' Miss.ZIPLNPCA_OneGradientDescent
#'
#' Estimation of the parameters and the missing data
#' @param Y count matrix
#' @param X covariates
#' @param q size of the latent space
#' @param params parameters including initial B
#' @param config configuration of the optimizer
#' @param tolS S tolerance
#' @return A list of the estimated parameters
#' @import PLNmodels
#' @export



Miss.ZIPLNPCA_OneGradientDescent <- function(Y, # Table de comptages n*p qui peut contenir des données manquantes
                          X, # Covariables np*d dont une colonne de 1 pour l'intercept
                          q, # Dimension de l'espace latent q
                          params,  # Paramètres fourni en entrée
                          method = "Only_B",
                          config = NULL,
                          tolS = NULL){
  
  n <- nrow(Y)
  p <- ncol(Y)
  d <- ncol(X)
  
  if (is.null(config)){config <- PLNPCA_param()$config_optim}
  if (is.null(tolS)){tolS <- 1e-04}
  
  lBound <- c(rep(-Inf, (2*d)+(p*q)+(n*q)), rep(tolS, n*q))
  config$lower_bounds <- lBound
  
  R <- ifelse(is.na(Y), 0, 1) # Masque qui met des 0 à la place des données manquantes
  
  Y.na <- ifelse(R == 0, 0, Y)
  
  data <- list(Y = Y.na,
               R = R,
               X = X)
  
  
  # out <- nlopt_optimize_ZIP_Beta(data, params, config)
  out <- switch(method,
         "Only_B" = nlopt_optimize_ZIP_Beta(data, params, config),
         "Only_D" = nlopt_optimize_ZIP_Gamma(data, params, config),
         "Only_C" = nlopt_optimize_ZIP_C(data, params, config),
         "Only_M" = nlopt_optimize_ZIP_M(data, params, config),
         "Only_S" = nlopt_optimize_ZIP_S(data, params, config),
         stop("Méthode d'optimisation invalide"))
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
              elbo1 = out$elbo1,
              elbo2 = out$elbo2,
              elbo3 = out$elbo3,
              elbo4 = out$elbo4,
              elbo5 = out$elbo5)
  
  return(res)
  
}
