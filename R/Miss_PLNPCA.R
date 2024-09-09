#' Miss.PLNPCA
#'
#' Estimation of the parameters and the missing data
#' @param Y count matrix
#' @param X covariates
#' @param q size of the latent space
#' @param params Initial parameters
#' @param config configuration of the optimizer
#' @param O offsets
#' @param w weights
#' @return A list of the estimated parameters
#' @import PLNmodels
#' @export





Miss.PLNPCA <- function(Y, # Table de comptages n*p qui peut contenir des données manquantes
                         X, # Covariables np*d dont une colonne de 1 pour l'intercept
                         O = NULL, # Offsets
                         w = NULL, # Poids
                         q, # Dimension de l'espace latent q
                         params = NULL, # Paramètres fourni en entrée
                         config = NULL){ # Paramètres pour l'optimisation

  n <- nrow(Y)
  p <- ncol(Y)

  if (is.null(params)){params <- Init(Y, X, q)}
  if (is.null(config)){config <- PLNPCA_param()$config_optim}
  if (is.null(O)){O <- matrix(0, nrow = n, ncol = p)}
  if (is.null(w)){w <- rep(1,n)}

  R <- ifelse(is.na(Y), 0, 1) # Masque qui met des 0 à la place des données manquantes

  Y.na <- ifelse(R == 0, 0, Y)

  data <- list(Y = Y.na,
               R = R,
               X = X,
               O = O,
               w = w)

  if (nrow(X)==n*p){
    out <- nlopt_optimize_rank_cov(data, params, config)
  }

  else {
    out <- nlopt_optimize_rank_miss(data, params, config)
  }

  mStep <- list(beta = out$B, C = out$C)
  eStep <- list(M = out$M, S = out$S)

  B.hat <- mStep$beta
  C.hat <- mStep$C
  M.hat <- eStep$M
  S.hat <- eStep$S
  XB.hat <- VectorToMatrix(X %*% B.hat, n, p)

  A <- O + XB.hat + M.hat %*% t(C.hat) + 0.5 * (S.hat * S.hat) %*% t(C.hat* C.hat)
  A <- exp(A)
  predicted <- exp(A)

  pred <- list(A = A, predicted = predicted)

  iter <- out$monitoring$iterations
  elboPath <- out$objective_values
  elbo <- out$objective

  res <- list(mStep = mStep,
              eStep = eStep,
              pred = pred,
              iter = iter,
              elboPath = elboPath,
              elbo = elbo,
              params.init = params,
              monitoring = out$monitoring)

  return(res)
}
