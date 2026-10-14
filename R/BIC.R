#' BIC from the ELBO for PLN-PCA and ZIP-PLN-PCA models
#'
#' Computes a BIC-like score in which the log-likelihood is replaced by the
#' ELBO of the fitted model (higher is better):
#' \deqn{\mathrm{BIC} = \mathrm{ELBO} - \frac{\log(n)}{2}\Big[q\{p - (q-1)/2\} + k\Big],}
#' where \eqn{q\{p - (q-1)/2\}} is the number of free parameters in \eqn{C}
#' (identifiable up to a rotation) and \eqn{k} the number of regression
#' parameters: \eqn{2d} for the ZIP-PLN-PCA model (\eqn{\beta} and
#' \eqn{\gamma}), \eqn{d} for the PLN-PCA model with a vectorized design
#' (\eqn{(np) \times d}) and \eqn{pd} with a rowwise design (\eqn{n \times d}).
#' \eqn{n} is the number of rows of \code{Y} (sites), which are the
#' independent units of the model.
#'
#' @param Y Numeric \code{n x p} count matrix (NAs allowed).
#' @param X Design matrix, \code{(n*p) x d} (or \code{n x d} for a rowwise
#'   PLN-PCA design).
#' @param fit Fitted object containing \code{$elbo}
#'   (output of \code{Miss.ZIPLNPCA} or \code{Miss.PLNPCA}).
#' @param q Integer, size of the latent space.
#' @param model \code{"ZIPLNPCA"} (default) or \code{"PLNPCA"}.
#'
#' @return A single numeric value, the BIC score.
#' @seealso \code{\link{ICL}}
#' @export
BIC <- function(Y, X, fit, q, model = c("ZIPLNPCA", "PLNPCA")){
  
  model <- match.arg(model)
  
  n <- nrow(Y)
  p <- ncol(Y)
  d <- ncol(X)
  elbo <- fit$elbo
  
  ## Number of regression parameters
  if (model == "ZIPLNPCA"){
    k <- 2*d                                    # beta and gamma
  } else {
    k <- if (nrow(X) == n*p) d else p*d         # vectorized or rowwise design
  }
  
  BIC <- elbo - (q*(p - (q-1)/2) + k)*log(n)/2
  
  return(BIC)
}


#' ICL from the ELBO for PLN-PCA and ZIP-PLN-PCA models
#'
#' Integrated Completed Likelihood: the BIC minus the entropy of the
#' variational distribution of the latent variables (higher is better):
#' \deqn{\mathrm{ICL} = \mathrm{BIC} - \mathcal{H}(W) - \mathcal{H}(U).}
#' \itemize{
#'   \item \eqn{\mathcal{H}(W)}: entropy of the Gaussian variational
#'     distribution of \eqn{W},
#'     \eqn{\frac{1}{2}\sum_{i,k} \log \sigma^2_{ik} + \frac{nq}{2}\{1 + \log(2\pi)\}},
#'     where \eqn{\sigma^2 = S} for the ZIP-PLN-PCA model (\code{S} is a
#'     variance) and \eqn{\sigma^2 = S^2} for the PLN-PCA model (\code{S} is a
#'     standard deviation). Zero when \code{q = 0}.
#'   \item \eqn{\mathcal{H}(U)}: entropy of the Bernoulli variational
#'     distribution of \eqn{U},
#'     \eqn{-\sum_{ij} \{\xi_{ij}\log\xi_{ij} + (1-\xi_{ij})\log(1-\xi_{ij})\}}
#'     (ZIP-PLN-PCA model only).
#' }
#'
#' @inheritParams BIC
#' @return A single numeric value, the ICL score.
#' @seealso \code{\link{BIC}}
#' @export
ICL <- function(Y, X, fit, q, model = c("ZIPLNPCA", "PLNPCA")){
  
  model <- match.arg(model)
  
  n <- nrow(Y)
  
  ## Entropy of W (Gaussian)
  if (q == 0){
    H_W <- 0
  } else {
    var_W <- if (model == "ZIPLNPCA") fit$eStep$S else fit$eStep$S^2
    H_W <- 0.5*sum(log(var_W)) + n*q/2*(1 + log(2*pi))
  }
  
  ## Entropy of U (Bernoulli), ZIP model only
  if (model == "ZIPLNPCA"){
    xi <- fit$eStep$xi
    H_U <- -sum(xi*log(xi) + (1 - xi)*log(1 - xi), na.rm = TRUE)  # 0 log 0 = 0
  } else {
    H_U <- 0
  }
  
  ICL <- BIC(Y, X, fit, q, model = model) - H_W - H_U
  
  return(ICL)
}