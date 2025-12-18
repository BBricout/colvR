#' ZI‑PLN-PCA with missing data using log(S) parametrization
#'
#' Fits a zero‑inflated Poisson log‑normal (ZI‑PLN-PCA) latent factor model to a
#' count matrix with missing values, using a variational objective and an
#' \strong{alternative parametrization} on the variational scales:
#' \deqn{ \code{log(S)} \text{ instead of } S. }
#' This guarantees positivity of \(S\) and allows simple box constraints on
#' \(\code{log(S)} \) via \code{tolLogS}.
#'
#' @param Y Numeric \code{n x p} count matrix. May contain \code{NA}.
#' @param X Numeric design matrix with \code{n*p} rows and \code{d} columns,
#'   aligned with \code{vec(Y)} (vectorization by columns).
#' @param q Integer, latent rank (dimension of the latent space).
#' @param params Optional list of initial parameters. If \code{NULL}, they are
#'   initialized by \code{\link{Init_ZIP}}. If provided, it must include a
#'   positive \code{S}; the function internally adds \code{logS <- log(S)}.
#' @param config Optional list of optimizer controls. If \code{NULL}, defaults
#'   to \code{PLNPCA_param()$config_optim}.
#' @param tolxi Numeric tolerance for the Jaakkola‑type \eqn{\xi} updates in the
#'   logistic bound (default \code{1e-4}).
#' @param tolLogS Numeric upper bound applied to \code{logS} (box constraint).
#'   Defaults to \code{Inf}. Use this to avoid excessively large variational
#'   scales when needed.
#'
#' @return A list with components:
#' \describe{
#'   \item{\code{mStep}}{Model parameters: \code{gamma} (\code{d x 1}),
#'     \code{beta} (\code{d x 1}), and loadings \code{C} (\code{p x q}).}
#'   \item{\code{eStep}}{Variational parameters: means \code{M} (\code{n x q}),
#'     scales \code{S} (\code{n x q}), and logistic bound parameters \code{xi}
#'     (\code{n x p}).}
#'   \item{\code{pred}}{List with predictors and expected counts:
#'     \code{mu} (\code{n x p}), \code{nu} (\code{n x p}), backend Poisson mean
#'     \code{A} (\code{n x p}), and \code{predicted} recomputed in R:
#'     \deqn{\exp\!\big( X B + M C^\top + 0.5\,(S\odot S)\,(C\odot C)^\top \big).}}
#'   \item{\code{imputed}}{\code{n x p} matrix equal to \code{xi * A} at missing
#'     entries of \code{Y}, and \code{Y} elsewhere.}
#'   \item{\code{iter}}{Number of iterations; \code{elboPath} (trajectory of ELBO);
#'     \code{elbo} (final ELBO).}
#'   \item{\code{params.init}}{Initial parameters passed to the backend (with
#'     \code{logS} added).}
#'   \item{\code{monitoring}}{Optimizer diagnostics/logs.}
#'   \item{\code{gradB, gradD, gradC, gradM, gradS}}{ELBO gradients w.r.t.
#'     corresponding parameters (via \code{\link{Elbo_grad}}).}
#' }
#'
#' @details
#' Missing entries are handled by a mask \code{R = 1_{observed}(Y)}; a working
#' matrix \code{Y.na} sets missings to 0 for the objective evaluation.
#' Box constraints are set on \code{logS} via \code{config$upper_bounds}, using
#' \code{tolLogS}. Make sure \code{X} matches \code{vec(Y)}.
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 40; p <- 12; d <- 3; q <- 2
#' Y <- matrix(rpois(n*p, 2), n, p); Y[sample(length(Y), 20)] <- NA
#' X <- cbind(1, rnorm(n*p), rnorm(n*p))  # (n*p) x d
#'
#' fit <- Miss.ZIPLNPCA.logS(Y = Y, X = X, q = q, tolLogS = 2)  # cap logS
#' fit$elbo
#' str(fit$eStep$S)
#' }
#'
#' @seealso \code{\link{Miss.ZIPLNPCA}} for the \code{S} parametrization,
#'   \code{\link{Init_ZIP}}, \code{\link{Elbo_grad}}
#' @import PLNmodels
#' @export



Miss.ZIPLNPCA.logS <- function(Y, # Table de comptages n*p qui peut contenir des données manquantes
                             X, # Covariables np*d dont une colonne de 1 pour l'intercept
                             q, # Dimension de l'espace latent q
                             params = NULL, # Paramètres fourni en entrée
                             config = NULL,
                             tolxi = NULL, 
                             tolLogS = NULL){

  n <- nrow(Y)
  p <- ncol(Y)
  d <- ncol(X)

  if (is.null(params)){params <- Init_ZIP(Y, X, q)}
  if (is.null(config)){config <- PLNPCA_param()$config_optim}
  if (is.null(tolxi)){tolxi <- 1e-04}
  if (is.null(tolLogS)){tolLogS <- Inf}
  
  uBound <- c(rep(Inf, (2*d)+(p*q)+(n*q)), rep(tolLogS, n*q))
  config$upper_bounds <- uBound

  R <- ifelse(is.na(Y), 0, 1) # Masque qui met des 0 à la place des données manquantes

  Y.na <- ifelse(R == 0, 0, Y)

  data <- list(Y = Y.na,
               R = R,
               X = X)

  params$logS <- log(params$S)

  out <- nlopt_optimize_ZIP_logS(data, params, config, tolxi)

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
  
  imputed <- ifelse(is.na(Y), out$xi*out$A, Y)
  grad <- Elbo_grad(data, params, tolxi)

  res <- list(mStep = mStep,
              eStep = eStep,
              pred = pred,
              imputed = imputed,
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
