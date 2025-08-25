#' ELBO and parameter gradients (ZIP‑PLN variational objective)
#'
#' Computes the Evidence Lower Bound (ELBO) and its gradients with respect to
#' the model and variational parameters for a zero‑inflated Poisson
#' log‑normal (ZIP‑PLN) latent factor model. This is a thin R wrapper around
#' a fast C++ implementation, with a small preprocessing step that replaces
#' \code{NA} values in \code{Y} by zeros before delegation.
#'
#' @param data A list with elements:
#'   \describe{
#'     \item{\code{Y}}{Numeric \code{n x p} count matrix (may contain \code{NA}).}
#'     \item{\code{X}}{Numeric \code{n x d} design matrix for the abundance/log‑mean part.}
#'     \item{\code{R}}{Numeric/binary design matrix for the zero‑inflation (logit) part
#'       (same number of rows as \code{Y}); can be \code{NULL} if not used.}
#'   }
#' @param params A list with elements:
#'   \describe{
#'     \item{\code{B}}{Matrix of regression coefficients for \code{X} (abundance).}
#'     \item{\code{D}}{Matrix of regression coefficients for \code{X} (zero‑inflation).}
#'     \item{\code{C}}{Loadings matrix for the latent factors (rank \code{q}).}
#'     \item{\code{M}}{Variational means of the latent factors.}
#'     \item{\code{S}}{Variational standard deviations (or log‑SDs) of the latent factors.}
#'   }
#' @param tolxi Numeric, convergence tolerance used for the variational
#'   bound on the logistic term.
#'
#' @return A list containing at least:
#'   \describe{
#'     \item{\code{elbo}}{Scalar numeric, the ELBO value.}
#'     \item{\code{grad_B, grad_D, grad_C, grad_M, grad_S}}{Gradients w.r.t. the corresponding parameters.}
#'   }
#'
#' @details
#' This function sets \code{NA} entries of \code{data$Y} to \code{0} internally
#' (they are not dropped) before calling the C++ backend
#' \code{Elbo_grad_Rcpp}. Ensure the dimensions of \code{X}, \code{R} and
#' the parameter matrices are compatible with \code{Y}.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5; d <- 2; r <- 2; q <- 2
#' Y <- matrix(rpois(n * p, 2), n, p); Y[sample(length(Y), 5)] <- NA
#' X <- cbind(1, rnorm(n))
#' R <- cbind(1, rnorm(n))   # design for zero-inflation
#' B <- matrix(0, d, p)
#' D <- matrix(0, r, p)
#' C <- matrix(rnorm(p * q, 0, .1), p, q)
#' M <- matrix(0, n, q)
#' S <- matrix(0.1, n, q)
#'
#' out <- Elbo_grad(
#'   data   = list(Y = Y, X = X, R = R),
#'   params = list(B = B, D = D, C = C, M = M, S = S),
#'   tolxi  = 1e-6
#' )
#' out$elbo
#'
#' @seealso \code{Elbo_grad_Rcpp} (backend C++).
#' @export
Elbo_grad <- function(data, params, tolxi) {
  Y.na <- ifelse(is.na(data$Y), 0, data$Y)
  data$Y <- Y.na
  return(Elbo_grad_Rcpp(data, params, tolxi))
}