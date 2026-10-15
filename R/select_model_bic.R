#' Model selection over candidate latent dimensions (BIC or ICL)
#'
#' For each value in \code{latents}, fits a model with \code{fun} and computes
#' the chosen criterion (\code{\link{BIC}} or \code{\link{ICL}}). Returns the
#' criterion values, the latent dimension that maximizes the criterion and the
#' corresponding fit.
#'
#' @param Y Numeric \code{n x p} count matrix (NAs allowed).
#' @param X Design matrix, \code{(n*p) x d}.
#' @param fun Fitting function, called as \code{fun(Y = Y, X = X, q = q)}
#'   (e.g. \code{Miss.ZIPLNPCA} or \code{Miss.PLNPCA}).
#' @param latents Integer vector of candidate latent dimensions.
#' @param model \code{"ZIPLNPCA"} (default) or \code{"PLNPCA"}; must match
#'   \code{fun}. Passed to \code{\link{BIC}} and \code{\link{ICL}}.
#' @param criterion \code{"BIC"} (default) or \code{"ICL"}.
#'
#' @details A progress bar is displayed via \pkg{pbapply}. The function checks
#' that \code{model} matches the fits returned by \code{fun}: a ZIP-PLN-PCA fit
#' contains \code{mStep$gamma}, a PLN-PCA fit does not.
#'
#' @return A list with elements:
#' \itemize{
#'   \item \code{criterion}: the criterion used.
#'   \item \code{values}: numeric vector of criterion values, named by
#'     \code{latents}.
#'   \item \code{q_star}: the latent dimension maximizing the criterion.
#'   \item \code{fit_star}: the fit obtained with \code{q_star}.
#' }
#'
#' @examples
#' set.seed(1)
#' n <- 30; p <- 5; d <- 2; q <- 2
#' X <- cbind(1, rnorm(n * p))                       # (n*p) x d design
#' theta <- list(B = matrix(c(1, 0.5)), D = matrix(c(1, -0.5)),
#'               C = matrix(rnorm(p * q, sd = 0.5), p, q))
#' Y <- Simul(X, theta, list(n = n, p = p, d = d, q = q))$Y
#' Y[sample(n * p, 20)] <- NA                        # missing entries
#'
#' \donttest{
#' sel <- select_model_bic(Y, X, Miss.ZIPLNPCA, latents = 1:3)
#' sel$values
#' sel$q_star
#' fit <- sel$fit_star
#'
#' sel_icl <- select_model_bic(Y, X, Miss.ZIPLNPCA, latents = 1:3,
#'                             criterion = "ICL")
#' sel_icl$q_star
#' }
#'
#' @seealso \code{\link{BIC}}, \code{\link{ICL}}
#' @importFrom pbapply pblapply
#' @export
select_model_bic <- function(Y, X, fun, latents,
                             model     = c("ZIPLNPCA", "PLNPCA"),
                             criterion = c("BIC", "ICL")){
  
  model <- match.arg(model)
  criterion <- match.arg(criterion)
  
  ## First fit, then consistency check between model and fun
  ## (checked before running the other fits)
  fit_1 <- fun(Y = Y, X = X, q = latents[1])
  is_zip <- !is.null(fit_1$mStep$gamma)
  if (is_zip != (model == "ZIPLNPCA"))
    stop("'model' does not match the fits returned by 'fun': ",
         if (is_zip) "fits are ZIPLNPCA, use model = \"ZIPLNPCA\"."
         else "fits are PLNPCA, use model = \"PLNPCA\".")
  
  ## Fits for the other values of q
  fits <- c(list(fit_1),
            pblapply(seq_along(latents)[-1], function(j)
              fun(Y = Y, X = X, q = latents[j])))
  
  ## Criterion
  crit_fun <- if (criterion == "BIC") BIC else ICL
  values <- vapply(seq_along(latents), function(j)
    crit_fun(Y, X, fits[[j]], latents[j], model = model), numeric(1))
  names(values) <- latents
  
  ## Best model
  j_star <- which.max(values)
  
  return(list(criterion = criterion,
              values    = values,
              q_star    = latents[j_star],
              fit_star  = fits[[j_star]]))
}