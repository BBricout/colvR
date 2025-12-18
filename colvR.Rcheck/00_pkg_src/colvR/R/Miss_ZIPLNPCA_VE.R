#' ZI‑PLN-PCA (missing data) — variational E‑step solver
#'
#' Fits a zero‑inflated Poisson log‑normal (ZI‑PLN-PCA) latent factor model on a
#' count matrix with missing values using a single call to a variational
#' optimization backend (VE). Returns model/variational parameters and
#' expected Poisson means suitable for imputation and prediction.
#'
#' @param Y Numeric \code{n x p} count matrix. May contain \code{NA}.
#' @param X Numeric design matrix with \code{n*p} rows and \code{d} columns,
#'   aligned with \code{vec(Y)} (vectorization by columns).
#' @param q Integer, latent rank (dimension of the latent space).
#' @param params Optional list of initial parameters. If \code{NULL},
#'   \code{\link{Init_ZIP}} is called.
#' @param config_vem List of outer VE controls (stopping and bounds), with fields:
#'   \describe{
#'     \item{\code{maxiter}}{Maximum iterations (not always used by this backend, default \code{10000}).}
#'     \item{\code{ftol}}{ELBO tolerance (default \code{1e-08}).}
#'     \item{\code{xtol}}{Parameter tolerance (default \code{1e-04}).}
#'     \item{\code{tolS}}{List with \code{lower}, \code{upper} bounds for \code{S}
#'       (default \code{list(lower = 1e-4, upper = 1)}).}
#'     \item{\code{tolxi}}{Tolerance for the Jaakkola‑type \eqn{\xi} updates (default \code{1e-04}).}
#'   }
#' @param config List of NLOpt controls for the backend (algorithm, tolerances,
#'   \code{maxeval}, etc.). If \code{NULL}, sensible defaults are provided.
#'
#' @return A list with components:
#' \describe{
#'   \item{\code{mStep}}{Model parameters: \code{gamma} (\code{d x 1}), \code{beta} (\code{d x 1}),
#'     loadings \code{C} (\code{p x q}).}
#'   \item{\code{eStep}}{Variational parameters: \code{M} (\code{n x q}),
#'     \code{S} (\code{n x q}), \code{xi} (\code{n x p}).}
#'   \item{\code{pred}}{List with predictors and expected counts:
#'     \code{mu} (\code{n x p}), \code{nu} (\code{n x p}), backend mean \code{A} (\code{n x p}),
#'     and \code{predicted} recomputed in R.}
#'   \item{\code{elbo}}{Final ELBO value.}
#'   \item{\code{params.init}}{Initial parameters used.}
#'   \item{\code{monitoring}}{Backend diagnostics (if provided).}
#'   \item{\code{elbo1, elbo2, elbo3, elbo4, elbo5}}{Decomposition terms of the ELBO (as returned by the backend).}
#' }
#'
#' @details
#' Missing entries are handled via a binary mask \code{R = 1_{observed}(Y)},
#' and a working copy \code{Y.na} with zeros at missing locations for the
#' objective evaluation. Box constraints on \code{S} are injected into
#' \code{config$lower_bounds} and \code{config$upper_bounds} using
#' \code{config_vem$tolS}.
#'
#' \strong{Predicted mean:} ici, la recomposition côté R utilise
#' \deqn{ \exp\!\big( X B + M C^\top + 0.5\, S \,(C\odot C)^\top \big). }
#' D'autres fonctions du package emploient
#' \eqn{0.5\,(S\odot S)\,(C\odot C)^\top}. Vérifie et harmonise selon la
#' paramétrisation attendue par ton backend.
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 40; p <- 12; d <- 3; q <- 2
#' Y <- matrix(rpois(n*p, 2), n, p); Y[sample(length(Y), 20)] <- NA
#' X <- cbind(1, rnorm(n*p), rnorm(n*p))  # (n*p) x d
#'
#' fit <- Miss.ZIPLNPCA_VE(Y = Y, X = X, q = q)
#' fit$elbo
#' str(fit$mStep); str(fit$eStep)
#' }
#'
#' @seealso \code{\link{Miss.ZIPLNPCA}}, \code{\link{Miss.ZIPLNPCA_Steps}},
#'   \code{\link{Init_ZIP}}
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