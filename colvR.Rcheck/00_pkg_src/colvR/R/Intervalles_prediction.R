#' Predictive simulations and intervals
#'
#' Draws parameter samples using the sandwich variance \code{V_theta}, runs VE
#' updates, and simulates predictive counts with zero-inflation. Returns
#' per-cell prediction intervals and coverage.
#'
#' @param Y An n x p count matrix (NAs allowed).
#' @param X An (n*p) x d design matrix (stacked by blocks of size n).
#' @param fit Estimator output list.
#' @param MC Integer, number of Monte Carlo draws.
#' @return A list with elements:
#' \itemize{
#'   \item \code{Z}, \code{U}, \code{Y.hat}, \code{pred}: MC samples
#'   \item \code{xi}: VE xi matrices for each draw
#'   \item \code{lower}, \code{upper}: 2.5\% and 97.5\% predictive quantiles
#'   \item \code{level}: average empirical coverage of \code{[lower, upper]}
#' }
#' @importFrom mvtnorm rmvnorm
#' @importFrom pbapply pblapply
#' @export
Predictions <- function(Y, X, fit, MC){
  
  ## Paramètres
  n <- nrow(Y) ; p <- ncol(Y) ; d <- ncol(X)
  B.hat <- fit$mStep$beta ; D.hat <- fit$mStep$gamma ; C.hat <- fit$mStep$C
  q <- ncol(C.hat)
  M.hat <- fit$eStep$M ; S.hat <- fit$eStep$S 
  xi <- fit$eStep$xi
  
  ## Variance sandwich 
  Var <- as.matrix(V_theta(Y, X, fit))
  
  ## Tirages de paramètres
  theta.hat <- c(as.vector(D.hat), as.vector(B.hat), as.vector(C.hat))
  theta.sample <- lapply(1:MC, function(m)
    rmvnorm(1, theta.hat, Var))
  
  gamma.sample <- lapply(1:MC, function(m)
    as.matrix(theta.sample[[m]][1:d]))
  beta.sample <- lapply(1:MC, function(m)
    as.matrix(theta.sample[[m]][(d+1):(2*d)]))
  C.sample <- lapply(1:MC, function(m)
    VectorToMatrix(theta.sample[[m]][(2*d+1):(2*d +p*q)], p, q))
  
  params.sample <- lapply(1:MC, function(m)
    list(B = matrix(beta.sample[[m]]), D = matrix(gamma.sample[[m]]), C = C.sample[[m]], M = M.hat, S = S.hat))
  
  ## Etapes VE
  VE.sample <- pblapply(1:MC, function(m)
    Miss.ZIPLNPCA_VE(Y, X, q, params = params.sample[[m]]))
  
  mu.sample <- lapply(1:MC, function(m)
    VectorToMatrix(X%*%beta.sample[[m]], n, p))
  
  nu.sample <- lapply(1:MC, function(m)
    VectorToMatrix(X%*%gamma.sample[[m]], n, p))
  
  A.VE <- pblapply(1:MC, function(m)
    exp(mu.sample[[m]] + VE.sample[[m]]$eStep$M%*%t(C.sample[[m]]) + 0.5 * VE.sample[[m]]$eStep$S %*% t(C.sample[[m]] * C.sample[[m]])))
  
  xi.VE <- lapply(1:MC, function(m)
    VE.sample[[m]]$eStep$xi)
  
  U <- lapply(1:MC, function(m)
    matrix(rbinom(n*p, prob = xi.VE[[m]], size = 1), nrow = n))
  
  mu <- lapply(1:MC, function(m)
    mu.sample[[m]] + VE.sample[[m]]$eStep$M %*% t(C.sample[[m]]))
  
  sigma <- lapply(1:MC, function(m){
    lapply(1:n, function(i)
      C.sample[[m]] %*% diag(VE.sample[[m]]$eStep$S[i,]) %*% t(C.sample[[m]]))
  })
  
  Z <- lapply(1:MC, function(m){do.call(rbind, lapply(1:n, function(i){rmvnorm(1, mu[[m]][i,], sigma[[m]][[i]])}))})
  Z_clipped <- lapply(Z, function(z) pmin(z, 700))
  
  Y.hat <- lapply(1:MC, function(m)
    matrix(rpois(n*p, lambda = exp(Z_clipped[[m]])), nrow = n))
  
  pred <- lapply(1:MC, function(m)
    Y.hat[[m]]*U[[m]])
  
  Y_sim_arr <- simplify2array(pred)
  
  Y_pred_lower <- apply(Y_sim_arr, c(1, 2), function(x) quantile(x, 0))
  Y_pred_upper <- apply(Y_sim_arr, c(1, 2), function(x) quantile(x, 0.9))
  
  
  res <- list(lower = Y_pred_lower, upper = Y_pred_upper)
  
  return(res)
  
}
