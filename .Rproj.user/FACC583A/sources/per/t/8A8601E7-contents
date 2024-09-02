#' Init_ZIP
#'
#' Parameters initialisation in the zero inflated case
#' @param Y count matrix
#' @param X covariates
#' @param q size of the latent space
#' @return A list with elements:
#'   \describe{
#'     \item{B}{Matrix of Poisson regression coefficients (d x p).}
#'     \item{D}{Matrix of logistic regression coefficients (d x p).}
#'     \item{C}{Matrix of latent structure estimates (p x q).}
#'     \item{M}{Matrix of variational parameters (n x p).}
#'     \item{S}{Matrix of variance parameters (n x q).}
#'   }
#' @export



Init_ZIP <- function(Y, X, q){

  n <- nrow(Y)
  p <- ncol(Y)
  vecY <- MatrixToVector(Y)

  U <- ifelse(Y == 0, 0, 1)
  vecU <- MatrixToVector(U)
  fit.logit <- glm(vecU ~ -1 + X, family = "binomial", na.action = na.exclude)
  D <- as.matrix(fit.logit$coefficients)

  fit <- lm(log(1 + vecY) ~ -1 + X, na.action = na.exclude)
  B <- as.matrix(fit$coefficients)
  res.mat <- VectorToMatrix(fit$residuals, n, p)

  svdM <- svd(res.mat, nu = q, nv = p)

  C <- svdM$v[, 1:q, drop = FALSE] %*% diag(svdM$d[1:q], nrow = q, ncol = q)/sqrt(n)
  M  <- svdM$u[, 1:q, drop = FALSE] %*% diag(svdM$d[1:q], nrow = q, ncol = q) %*% t(svdM$v[1:q, 1:q, drop = FALSE])
  S <- matrix(1, n, q)

  return(list(B = B, D = D, C = C, M = M, S = S))
}
