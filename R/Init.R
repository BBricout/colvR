#' Initialize parameters for a PLN-PCA with missing data
#'
#' Provides a quick, SVD‑based initialization of model and variational
#' parameters from a (possibly sparse) count matrix \code{Y} and a design
#' matrix \code{X}. The routine fits a (log‑transformed) linear model to
#' obtain regression coefficients and uses the SVD of residuals to
#' initialize the latent structure (loadings and variational means).
#'
#'
#' Missing values in \code{Y} are handled via \code{na.exclude} in the linear
#' model; residuals at missing entries are then set to zero before the SVD.
#'
#' @param Y Numeric \code{n x p} count matrix (may contain \code{NA}).
#' @param X Numeric design matrix of covariates. Either \code{n x d}
#'   (rowwise design) or \code{(n*p) x d} (vectorized design matching
#'   \code{MatrixToVector(Y)}).
#' @param q Integer, target latent dimension (rank) for the factor structure.
#'
#' @return A list with elements:
#' \describe{
#'   \item{\code{B}}{Matrix of regression coefficients (dimensions follow the
#'   chosen design; typically (\code{1 x d}) in the rowwise case).}
#'   \item{\code{C}}{Loadings matrix (\code{p x q}) for the latent factors.}
#'   \item{\code{M}}{Variational means of the latent factors (\code{n x q}).}
#'   \item{\code{S}}{Variational standard deviations (initialized to a small
#'   constant; \code{n x q}).}
#' }
#'
#'
#' @examples
#' set.seed(1)
#' n <- 30; p <- 8; d <- 2; q <- 2
#' Y <- matrix(rpois(n * p, 2), n, p)
#' X <- cbind(1, rnorm(n))          # rowwise design
#' init <- Init(Y, X, q)
#' str(init)
#'
#' @seealso \code{\link{MatrixToVector}}, \code{\link{VectorToMatrix}}
#' @export




Init <- function(Y, X, q){

  n <- nrow(Y)
  p <- ncol(Y)
  vecY <- MatrixToVector(Y)

  if (nrow(X)==n*p){
    fit <- lm(log(1 + vecY) ~ -1 + X, na.action = na.exclude)

    B <- as.matrix(fit$coefficients)
    res.vec <- fit$residuals
    res.full <- ifelse(is.na(vecY), 0, res.vec)

    res.mat <- VectorToMatrix(res.full, n, p)}

  else {
    fit <- lm(log(1 + Y) ~ -1 + X, na.action = na.exclude)

    B <- as.matrix(fit$coefficients)
    res <- fit$residuals
    res.full <- ifelse(is.na(Y), 0, res)
    res.mat <- res.full

  }

  svdM <- svd(res.mat, nu = q, nv = p)

  C <- svdM$v[, 1:q, drop = FALSE] %*% diag(svdM$d[1:q], nrow = q, ncol = q)/sqrt(n)
  M  <- svdM$u[, 1:q, drop = FALSE] %*% diag(svdM$d[1:q], nrow = q, ncol = q) %*% t(svdM$v[1:q, 1:q, drop = FALSE])
  S <- matrix(0.1, n, q)

  return(list(B = B, C = C, M = M, S = S))
}
