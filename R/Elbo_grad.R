#' Calculate the Elbo and the gradients
#'
#'
#' @param data list(Y, R, X)
#' @param params list(B, D, C, M, S)
#' @return Elbo and gradients
#' @export
Elbo_grad <- function(data, params, tolxi) {
  Y.na <- ifelse(is.na(data$Y), 0, data$Y)
  data$Y <- Y.na
  return(Elbo_grad_Rcpp(data, params, tolxi))
}