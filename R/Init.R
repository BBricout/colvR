#' Init
#'
#' Parameters initialisation
#' @param Y count matrix
#' @param X covariates
#' @param q size of the latent space
#' @return A list with elements:
#'   \describe{
#'     \item{B}{Matrix of Poisson regression coefficients (d x p).}
#'     \item{C}{Matrix of latent structure estimates (p x q).}
#'     \item{M}{Matrix of variational parameters (n x p).}
#'     \item{S}{Matrix of variance parameters (n x q).}
#'   }
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