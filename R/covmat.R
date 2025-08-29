#' Wrapper pour covmat du package lori
#'
#' Cette fonction appelle directement \code{lori::covmat}. 
#' Elle est fournie ici pour éviter à l'utilisateur de charger lori.
#'
#' @inheritParams lori::covmat
#' @return La matrice de covariables telle que calculée par \code{lori::covmat}.
#' @export
#' @examples
#' if (requireNamespace("lori", quietly = TRUE)) {
#'   n <- 3; p <- 2
#'   R <- matrix(rnorm(n*2), n)
#'   C <- matrix(rnorm(p*2), p)
#'   X <- covmat(n, p, R, C)
#'   dim(X)  # 6 x 4
#' }
covmat <- function(n, p, R = NULL, C = NULL, E = NULL, center = FALSE) {
  lori::covmat(n = n, p = p, R = R, C = C, E = E, center = center)
}
