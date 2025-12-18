#' Wrapper for `lori::covmat`
#'
#' Thin helper that calls \code{lori::covmat()} without requiring users to
#' attach the \pkg{lori} package. It constructs the block-structured
#' covariate matrix \eqn{X} (with \eqn{n \times p} rows) from site-level
#' covariates \code{R}, year-level covariates \code{C}, and optional site–year
#' covariates \code{E}. Any of \code{R}, \code{C}, or \code{E} can be \code{NULL}
#' to omit that block.
#'
#' @param n Integer; number of sites.
#' @param p Integer; number of years.
#' @param R Optional numeric matrix with \eqn{n} rows and \eqn{d_R} columns
#'   (site-level covariates). Default: \code{NULL}.
#' @param C Optional numeric matrix with \eqn{p} rows and \eqn{d_C} columns
#'   (year-level covariates). Default: \code{NULL}.
#' @param E Optional numeric matrix with \eqn{n p} rows and \eqn{d_E} columns
#'   (site–year covariates; one row per site–year observation in the same order
#'   as \eqn{X}). Default: \code{NULL}.
#' @param center Logical; whether to center columns within each block (as in
#'   \code{lori::covmat}). Default: \code{FALSE}.
#'
#' @return A numeric matrix with \eqn{n \times p} rows and
#'   \eqn{d = d_R + d_C + d_E} columns (when present), identical to the value
#'   returned by \code{lori::covmat()}.
#' @seealso \code{\link[lori]{covmat}}
#' @export
#' @examples
#' if (requireNamespace("lori", quietly = TRUE)) {
#'   set.seed(1)
#'   n <- 3; p <- 2
#'   R <- matrix(rnorm(n * 2), nrow = n, ncol = 2)  # site-level (d_R = 2)
#'   C <- matrix(rnorm(p * 2), nrow = p, ncol = 2)  # year-level (d_C = 2)
#'   X <- covmat(n, p, R = R, C = C)
#'   dim(X)  # 6 x 4
#' }
covmat <- function(n, p, R = NULL, C = NULL, E = NULL, center = FALSE) {
  lori::covmat(n = n, p = p, R = R, C = C, E = E, center = center)
}

covmat <- function(n, p, R = NULL, C = NULL, E = NULL, center = FALSE) {
  lori::covmat(n = n, p = p, R = R, C = C, E = E, center = center)
}

covmat <- function(n, p, R = NULL, C = NULL, E = NULL, center = FALSE) {
  lori::covmat(n = n, p = p, R = R, C = C, E = E, center = center)
}

