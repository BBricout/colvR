#' Initialize parameters for the ZIP case (zero-inflated)
#'
#' SVD-based initialization of parameters for a zero-inflated Poisson
#' log-normal latent factor model (ZIP-PLN). Regression coefficients for
#' the abundance part (\code{B}) are obtained by regressing \code{log(1+vec(Y))}
#' on \code{X}; zero-inflation coefficients (\code{D}) come from a logistic
#' regression of \code{I(Y>0)} on \code{X}. The latent structure is initialized
#' from the SVD of the residual matrix.
#'
#' @param Y Numeric \code{n x p} count matrix (may contain \code{NA}).
#' @param X Numeric design matrix with \code{n*p} rows and \code{d} columns,
#'   aligned with \code{vec(Y)} (column-wise vectorization).
#' @param q Integer, target latent dimension (rank).
#'
#' @return A list with elements:
#' \describe{
#'   \item{\code{B}}{Abundance (Poisson) regression coefficients (\code{1 x d}).}
#'   \item{\code{D}}{Zero-inflation (logit) regression coefficients (\code{1 x d}).}
#'   \item{\code{C}}{Loadings matrix (\code{p x q}).}
#'   \item{\code{M}}{Variational means of latent factors (\code{n x q}).}
#'   \item{\code{S}}{Variational scale parameters.}
#' }
#'
#'
#' @examples
#' set.seed(1)
#' n <- 30; p <- 10; d <- 3; q <- 2
#' Y <- matrix(rpois(n*p, 2), n, p)
#' X <- cbind(1, rnorm(n*p), rnorm(n*p))  # (n*p) x d, vectorized design
#' init <- Init_ZIP(Y, X, q)
#' str(init)
#'
#' @seealso \code{\link{Init}} for the non-ZI initializer;
#'   \code{\link{MatrixToVector}}, \code{\link{VectorToMatrix}}
#' @export




Init_ZIP <- function(Y, X, q){

  n <- nrow(Y) ; p <- ncol(Y)
  vecY <- MatrixToVector(Y)
  
  fit <- lm(log(1 + vecY) ~ -1 + X, na.action = na.exclude)
  B <- as.matrix(fit$coefficients)
  res.vec <- ifelse(is.na(vecY), 0, fit$residuals)
  res.mat <- VectorToMatrix(res.vec, n, p)
  
  U <- ifelse(Y == 0, 0, 1)
  vecU <- MatrixToVector(U)
  fit.logit <- glm(vecU ~ -1 + X, family = "binomial", na.action = na.exclude)
  D <- as.matrix(fit.logit$coefficients)
  
  svdM <- svd(res.mat, nu = q, nv = p)
  
  C <- svdM$v[, 1:q, drop = FALSE] %*% diag(svdM$d[1:q], nrow = q, ncol = q)/sqrt(n)
  M  <- svdM$u[, 1:q, drop = FALSE] %*% diag(svdM$d[1:q], nrow = q, ncol = q) %*% t(svdM$v[1:q, 1:q, drop = FALSE])
  Sigma <- C%*%t(C) + summary(fit)$sigma**2 * diag(p)
  diag(q) - t(C)%*% solve(Sigma) %*% C
  S <- rep(1,n) %o% diag(diag(q) - t(C)%*% solve(Sigma) %*% C)

  return(list(B = B, D = D, C = C, M = M, S = S))
}
