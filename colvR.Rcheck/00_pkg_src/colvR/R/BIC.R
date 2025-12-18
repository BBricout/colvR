#' BIC from ELBO for a ZIP-PLN latent factor model
#'
#' Computes a Bayesian Information Criterion (BIC)-like score from the
#' Evidence Lower Bound (ELBO) of a fitted model. The penalty accounts for
#' the number of free parameters in a latent factor model with latent
#' dimension \code{q}:
#' \deqn{ \mathrm{BIC} = \mathrm{ELBO} - \frac{\log(n)}{2}\Big[q\{p - (q-1)/2\} + p\,d\Big], }
#' where \eqn{n} is the number of rows (sites/samples), \eqn{p} the number of
#' columns (species/variables), and \eqn{d} the number of covariates.
#'
#' @param Y Numeric \code{n x p} count matrix.
#' @param X Numeric \code{n x d} design matrix of covariates (first column
#'   typically an intercept).
#' @param fit A fitted object (list) containing at least \code{$elbo}, the
#'   maximized ELBO value for the model.
#' @param q Integer, size of the latent space (rank).
#'
#' @return A single numeric value: the BIC score (higher is better in this
#'   ELBO-based convention).
#'
#' @details
#' This criterion mirrors the usual BIC structure but replaces the
#' log-likelihood with the ELBO and uses the parameter count appropriate
#' for a rank-\code{q} latent factor structure.
#'
#'
#' @seealso \code{\link[stats]{BIC}}
#' @export

BIC <- function(Y, X, fit, q){
  
  n <- nrow(Y)
  p <- ncol(Y)
  d <- ncol(X)
  elbo <- fit$elbo
  # BIC <- Jq - (p*(d+q)*log(n))/2
  BIC <- elbo - (q*(p - (q-1)/2) + p*d)*log(n)/2
  
  return(BIC)
}