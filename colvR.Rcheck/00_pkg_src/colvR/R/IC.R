#' 95\% interval for a theta
#' 95\% confidence interval for theta
#'
#' Computes the 95\% confidence interval for \code{theta}.
#'
#' @param theta Numeric vector of estimates.
#' @param var Numeric vector of associated variances (same length as \code{theta}).
#' @return A numeric vector of length 2: lower and upper bounds.
#' @examples
#' IC(1, 0.04)
#' @export
IC <- function(theta, var){
  c(theta - 1.96*sqrt(var), theta + 1.96*sqrt(var))
}
