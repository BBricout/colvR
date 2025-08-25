#' ZI‑PLN-PCA with missing data (variational fit + imputation)
#'
#' Fits a zero‑inflated Poisson log‑normal (ZI‑PLN-PCA) latent factor model to a
#' count matrix with missing values. The routine optimizes a variational
#' objective with NLOpt backends and returns parameter estimates, variational
#' parameters and an imputed matrix.
#'
#' This function supports latent rank \code{q >= 0}. When \code{q = 0}, it
#' reduces to a ZIP regression (no latent factors). When \code{q > 0}, it fits
#' a ZIP‑PLN factor model.
#'
#' @param Y Numeric \code{n x p} count matrix. May contain \code{NA}.
#' @param X Numeric design matrix with \code{n*p} rows and \code{d} columns,
#'   aligned with \code{vec(Y)} (column‑wise vectorization).
#' @param q Integer, latent rank (dimension of the latent space).
#' @param params Optional list of initial parameters. If \code{NULL}, they are
#'   initialized internally via \code{\link{Init_ZIP_q0}} (when \code{q = 0})
#'   or \code{\link{Init_ZIP}} (when \code{q > 0}).
#' @param config Optional list of optimizer controls. If \code{NULL}, a default
#'   configuration is used (NLOpt backend, MMA algorithm, sensible tolerances).
#' @param tolS List with numeric bounds for the variational scales \code{S}:
#'   elements \code{lower} and \code{upper}. Defaults to \code{list(lower = 0, upper = 1)}.
#' @param tolxi Numeric tolerance for the Jaakkola‑type \eqn{\xi} updates in the
#'   logistic bound (default \code{1e-4}).
#'
#' @return A list with components:
#' \describe{
#'   \item{\code{mStep}}{List of model parameters. For \code{q = 0}:
#'     \code{gamma} (\code{d x 1}) and \code{beta} (\code{d x 1}).
#'     For \code{q > 0}: same plus the loading matrix \code{C} (\code{p x q}).}
#'   \item{\code{eStep}}{List of variational parameters. For \code{q = 0}:
#'     only \code{xi} (\code{n x p}). For \code{q > 0}: \code{M} (\code{n x q}),
#'     \code{S} (\code{n x q}) and \code{xi} (\code{n x p}).}
#'   \item{\code{pred}}{List with predictors and expected counts:
#'     \code{mu} (\code{n x p}, abundance mean),
#'     \code{nu} (\code{n x p}, zero‑inflation mean),
#'     \code{A} (\code{n x p}, PLN expectation),
#'     \code{predicted} (\code{n x p}, predicted values).
#'     For \code{q = 0}, \code{predicted = exp(XB)}; for \code{q > 0},
#'     \deqn{ \mathrm{predicted} = \exp\!\big( XB + M C^\top + 0.5\,(S\odot S)\,(C\odot C)^\top \big). }}
#'   \item{\code{imputed}}{\code{n x p} matrix equal to \code{xi * A} at
#'     missing entries of \code{Y}, and \code{Y} elsewhere (ZIP expectation).}
#'   \item{\code{iter}}{Integer, number of iterations.}
#'   \item{\code{elboPath}}{Numeric vector of objective (ELBO) values over iterations.}
#'   \item{\code{elbo}}{Final ELBO value.}
#'   \item{\code{params.init}}{Parameters as recorded from the backend call.}
#'   \item{\code{monitoring}}{Optimizer diagnostics/logs from the backend.}
#'   \item{\code{gradB, gradD, gradC, gradM, gradS}}{Gradients of the ELBO with
#'     respect to the corresponding parameters, obtained via \code{\link{Elbo_grad}}.}
#' }
#'
#' @details
#' Internally, a binary mask \code{R = 1_{observed}(Y)} is created; a copied
#' matrix \code{Y.na} sets missing entries to zero for objective evaluation.
#' The optimizer bounds for \code{S} are taken from \code{tolS}. Ensure that
#' the vectorization and parameter stacking used in the optimizer are
#' consistent with the shapes listed above.
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 40; p <- 12; d <- 3
#' q <- 2
#' Y <- matrix(rpois(n*p, 2), n, p)
#' Y[sample(length(Y), 25)] <- NA
#' # Vectorized design (n*p) x d:
#' X <- cbind(1, rnorm(n*p), rnorm(n*p))
#'
#' fit <- Miss.ZIPLNPCA(Y = Y, X = X, q = q)
#' str(fit$mStep)
#' str(fit$eStep)
#' image(log1p(fit$imputed))  # quick look at imputed counts
#'
#' # Rank-0 (ZIP regression without latent factors):
#' fit0 <- Miss.ZIPLNPCA(Y = Y, X = X, q = 0)
#' fit0$eStep$xi[1:3, 1:3]
#' }
#'
#' @seealso \code{\link{Init_ZIP}}, \code{\link{Init_ZIP_q0}},
#'   \code{\link{Elbo_grad}}, \code{\link{VectorToMatrix}}
#' @import PLNmodels
#' @export




Miss.ZIPLNPCA <- function(Y, # Table de comptages n*p qui peut contenir des données manquantes
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
  if (is.null(tolS)){tolS <- list(lower = 0, upper = 1)}
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
  
  if(q == 0){
    
    if (is.null(params)){params <- Init_ZIP_q0(Y, X, q)}
    
    out <- nlopt_optimize_ZIP_q0(data, params, config, tolxi)
    mu <- VectorToMatrix(X%*%out$B, n, p)
    nu <- VectorToMatrix(X%*%out$D, n, p)
    
    mStep <- list(gamma = out$D, beta = out$B)
    eStep <- list(xi = out$xi)
    
    B.hat <- mStep$beta
    D.hat <- mStep$gamma
    XB.hat <- VectorToMatrix(X %*% B.hat, n, p)
    XD.hat <- VectorToMatrix(X %*% D.hat, n, p)
    
    predicted <- exp(XB.hat)
    elbo1 <- out$elbo1 ; elbo2 <- 0 ; elbo3 <- out$elbo3
    elbo4 <- out$elbo4 ; elbo5 <- 0
  }
  
  else{
    if (is.null(params)){params <- Init_ZIP(Y, X, q)}
    
    out <- nlopt_optimize_ZIP(data, params, config, tolxi)
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
  }
  
  params <- list(B = out$B, D = out$D, C = out$C, M = out$M, S = out$S)
  
  grad <- Elbo_grad(data, params, tolxi)

  pred <- list(A = out$A, nu = nu, mu = mu, predicted = predicted)
  iter <- out$monitoring$iterations
  elboPath <- out$objective_values
  # elbo <- out$objective_values[length(out$objective_values)]
  elbo <- out$objective
  imputed <- ifelse(is.na(Y), out$xi*out$A, Y)


  res <- list(mStep = mStep,
              eStep = eStep,
              imputed = imputed,
              pred = pred,
              iter = iter,
              elboPath = elboPath,
              elbo = elbo,
              params.init = params,
              monitoring = out$monitoring,
              gradB = grad$gradB,
              gradD = grad$gradD,
              gradC = grad$gradC,
              gradM = grad$gradM, 
              gradS = grad$gradS
                )

  return(res)

}
