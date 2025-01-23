#' BIC
#'
#' Parameters initialisation
#' @param Y count matrix
#' @param X covariates
#' @param fit fitting
#' @param q size of the latent space
#' @return A list with elements:
#'   \describe{
#'    \item{BIC} Bayesian information criterion
#'   }
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