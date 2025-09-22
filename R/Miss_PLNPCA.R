#' PLN-PCA with missing data (variational fit + imputation)
#'
#' Fits a Poisson log-normal PCA model (PLN-PCA) to a count matrix with
#' missing values, using a variational objective optimized by NLOpt-based
#' routines. The function estimates model/variational parameters and returns
#' expected counts to impute missing entries.
#'
#' Two designs for \code{X} are supported:
#' \enumerate{
#'   \item \strong{Rowwise design} (\code{nrow(X) == n}): one row per sample.
#'   \item \strong{Vectorized design} (\code{nrow(X) == n*p}): \code{X} aligns with
#'   \code{vec(Y)} (vectorization by columns).
#' }
#'
#'
#' @param Y Numeric \code{n x p} count matrix. May contain \code{NA}.
#' @param X Numeric design matrix of covariates:
#'   either \code{n x d} (rowwise) or \code{(n*p) x d} (vectorized to match
#'   \code{MatrixToVector(Y)}).
#' @param q Integer, latent rank (dimension of the latent space).
#' @param params Optional initial parameters (list) typically produced by
#'   \code{\link{Init}}; if \code{NULL}, they are initialized internally.
#' @param config Optional optimizer configuration; if \code{NULL}, defaults to
#'   \code{PLNPCA_param()$config_optim}.
#' @param O Optional numeric \code{n x p} matrix of offsets (default: zeros).
#' @param w Optional numeric vector of length \code{n} with observation weights
#'   (default: all ones).
#'
#' @return A list with components:
#' \describe{
#'   \item{\code{mStep}}{List with \code{beta} (\code{1 x d}) and \code{C} (\code{p x q}).}
#'   \item{\code{eStep}}{List with \code{M} (\code{n x q}) and \code{S} (\code{n x q}).}
#'   \item{\code{pred}}{List with
#'     \code{A} and \code{predicted}, both equal to the \code{n x p} matrix of
#'     expected counts used for imputation:
#'     \deqn{ A = \exp\!\big(O + X B + M C^\top + 0.5\,(S\odot S)\,(C\odot C)^\top\big), }
#'     where \eqn{\odot} denotes the Hadamard product and \code{X B} is reshaped to
#'     \code{n x p} when \code{X} is vectorized.}
#'   \item{\code{iter}}{Integer, number of iterations.}
#'   \item{\code{elboPath}}{Numeric vector of objective (ELBO) values over iterations.}
#'   \item{\code{elbo}}{Final ELBO value.}
#'   \item{\code{params.init}}{The initial parameters used.}
#'   \item{\code{monitoring}}{Optimizer log/diagnostics as returned by the backend.}
#' }
#'
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 40; p <- 12; d <- 2; q <- 2
#' Y <- matrix(rpois(n*p, 2), n, p)
#' Y[sample(length(Y), 20)] <- NA
#' X <- cbind(1, rnorm(n))      # rowwise design
#'
#' fit <- Miss.PLNPCA(
#'   Y = Y, X = X, q = q,
#'   O = matrix(0, n, p),
#'   w = rep(1, n),
#'   params = NULL,
#'   config = NULL
#' )
#' str(fit$pred$A)     # expected counts (n x p)
#' fit$elbo            # final ELBO
#' }
#'
#' @seealso \code{\link{Init}}, \code{\link{PLNPCA_param}},
#'   \code{\link{MatrixToVector}}, \code{\link{VectorToMatrix}},
#'   \code{\link{Elbo_grad}}
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
  predicted <- A

  pred <- list(A = A, predicted = predicted)

  iter <- out$monitoring$iterations
  elboPath <- out$objective_values
  elbo <- -out$objective

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
