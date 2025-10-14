#' Prediction intervals (marginal) for ZIPLNPCA-like model
#'
#' Computes cell-wise predictive simulations and 95% prediction intervals
#' under the *marginal* model (integrating latent factors).
#'
#' @param Y Matrix \code{n x p} of observed counts.
#' @param X Design matrix of shape \code{(n*p) x d} (your current design), or
#'   any object compatible with \code{as.numeric(X \%*\% beta)} producing a vector of length \code{n*p}
#'   later reshaped into \code{n x p}.
#' @param fit A fitted object providing \code{fit$mStep$beta}, \code{fit$mStep$gamma},
#'   \code{fit$mStep$C}, and \code{fit$eStep$M}, \code{fit$eStep$S}, \code{fit$eStep$xi}.
#'   Only \code{beta}, \code{gamma}, \code{C} are used here.
#' @param MC Integer, number of Monte Carlo draws.
#'
#' @details
#' The parameter vector is sampled with the sandwich variance from
#' \code{V_theta(Y, X, fit)}. The ordering assumed for \code{theta} is
#' \code{c(vec(gamma), vec(beta), vec(C))}, consistent with your code.
#'
#' Zero-inflation convention used here:
#' \deqn{\pi_0 = \mathrm{logit}^{-1}(X\gamma)} is the probability of a structural zero.
#' We simulate \eqn{U = \mathbf{1}\{\text{keep Poisson}\} \sim \mathrm{Bernoulli}(1-\pi_0)},
#' and the final prediction is \eqn{Y = U \cdot Y_{\mathrm{Poisson}}}.
#'
#' @return A list with elements:
#' \itemize{
#'   \item \code{Z}: list of length \code{MC} with Gaussian latent draws (n x p).
#'   \item \code{U}: list of length \code{MC} with Bernoulli “keep” masks (n x p).
#'   \item \code{Y.hat}: list of length \code{MC} with Poisson draws (n x p).
#'   \item \code{pred}: list of length \code{MC} with final ZIP predictions (n x p).
#'   \item \code{lower}, \code{upper}: matrices \code{n x p} with 2.5% and 97.5% PIs.
#'   \item \code{level}: scalar, empirical coverage vs \code{Y} (if used as “truth”).
#' }
#'
#' @examples
#' \dontrun{
#' n <- 5; p <- 3; d <- 2
#' set.seed(1)
#' Y <- matrix(rpois(n*p, 2), n, p)
#' X <- matrix(rnorm(n*p*d), n*p, d)
#' fit <- list(
#'   mStep = list(beta = rnorm(d), gamma = rnorm(d), C = matrix(rnorm(p*2), p, 2)),
#'   eStep = list(M = matrix(0, n, 2), S = matrix(1, n, 2), xi = matrix(0.5, n, p))
#' )
#' out <- Predictions.marginales(Y, X, fit, MC = 100)
#' str(out$lower)
#' }
#'
#' @seealso \code{\link{V_theta}}, \code{\link{VectorToMatrix}}
#' @export
#' @importFrom mvtnorm rmvnorm
#' @importFrom stats rbinom rpois quantile
#' @importFrom pbapply pblapply

Predictions.marginales <- function(Y, X, fit, MC){
  n <- nrow(Y) ; p <- ncol(Y) ; d <- ncol(X)
  
  # Parameters
  B.hat <- fit$mStep$beta ; D.hat <- fit$mStep$gamma ; C.hat <- fit$mStep$C
  q <- ncol(C.hat)
  M.hat <- fit$eStep$M ; S.hat <- fit$eStep$S 
  xi <- fit$eStep$xi
  
  # Variance sandwich 
  Var <- as.matrix(V_theta(Y, X, fit))
  
  # Simulation des paramètres (Sandwich)
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
  
  
  #Simulations des Y et U
  
  nu <- lapply(1:MC, function(m)
    VectorToMatrix(X%*%gamma.sample[[m]], n, p))
  pi <- lapply(1:MC, function(m)
    plogis(nu[[m]]))
  
  U <- lapply(1:MC, function(m)
    matrix(rbinom(n*p, p = pi[[m]], size = 1), nrow = n))
  
  mu <- lapply(1:MC, function(m)
    VectorToMatrix(X %*% matrix(beta.sample[[m]]), n, p))
  sigma <- lapply(1:MC, function(m)
    C.sample[[m]]%*%t(C.sample[[m]]))
  
  # Z et Y
  
  Z <- lapply(1:MC, function(m){do.call(rbind, lapply(1:n, function(i){rmvnorm(1, mu[[m]][i,], sigma[[m]])}))})
  Z_clipped <- lapply(Z, function(z) pmin(z, 700))
  
  Y.hat <- lapply(1:MC, function(m)
    matrix(rpois(n*p, lambda = exp(Z_clipped[[m]])), nrow = n))
  
  pred <- lapply(1:MC, function(m)
    Y.hat[[m]]*U[[m]])
  
  Y_sim_arr <- simplify2array(pred)
  
  Y_pred_lower <- apply(Y_sim_arr, c(1, 2), function(x) quantile(x, 0.025))
  Y_pred_upper <- apply(Y_sim_arr, c(1, 2), function(x) quantile(x, 0.975))
  
  level <- mean((Y - Y_pred_lower)*(Y_pred_upper - Y) >= 0, na.rm = TRUE)
  
  res <- list(Z = Z, U = U, Y.hat = Y.hat, pred = pred,
              lower = Y_pred_lower, upper = Y_pred_upper, level = level)
  
  return(res)
}
















