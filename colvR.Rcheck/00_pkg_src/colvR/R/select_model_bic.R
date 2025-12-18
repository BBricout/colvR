#' Model selection by BIC over candidate latent dimensions
#'
#' For each value in \code{latents}, fits a model via \code{fun(Y, X, q)}
#' and computes the corresponding BIC. Returns the vector of BIC values
#' in the same order as \code{latents}.
#'
#' @param Y Response matrix (n x p).
#' @param X Covariate matrix (n x d).
#' @param fun Fitting function called as \code{fun(Y, X, q)} that returns a fit object.
#' @param latents Integer/numeric vector of candidate latent dimensions.
#'
#' @details A progress bar is displayed via \pkg{pbapply}. The \code{BIC}
#' function used must be available in your namespace (e.g., \code{colvR::BIC}
#' if you define one, or \code{stats::BIC} if you call it explicitly).
#'
#' @return A numeric vector of BIC values, one per element of \code{latents},
#' in the same order as \code{latents}.
#'
#'
#' @importFrom pbapply pblapply
#' @export



select_model_bic <- function(Y, X, fun, latents){
  fits <- pblapply(latents, function(j)
    fun(Y = Y, X = X, q = latents[j]))
  bic <- unlist(lapply(latents, function(j)
    BIC(Y, X, fits[[j]], latents[[j]])))
  return(bic)
}