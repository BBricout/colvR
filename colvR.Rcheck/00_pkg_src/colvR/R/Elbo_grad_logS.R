#' Calculate the Elbo with log(S) instead of S and the gradients
#'
#'
#' @param data list(Y, R, X)
#' @param params list(B, D, C, M, S)
#' @return Elbo and gradients
#' @export
Elbo_grad_logS <- function(data, params) {
  return(Elbo_grad_logS_Rcpp(data, params))
}