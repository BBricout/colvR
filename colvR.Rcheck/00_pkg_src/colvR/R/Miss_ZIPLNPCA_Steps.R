#' ZI‑PLN-PCA (missing data) with parameterwise optimization steps
#'
#' Fits a zero‑inflated Poisson log‑normal (ZI‑PLN-PCA) latent factor model to a
#' count matrix with missing values using a \strong{blockwise} optimization
#' strategy. At each outer iteration, parameters are updated by activating
#' one block among \{B, D, C, M, S\} while keeping the others fixed, via an
#' NLOpt backend. The loop stops when both parameter and ELBO changes are below
#' thresholds or when \code{maxiter} is reached.
#'
#' Missing entries are handled through a binary mask and a working copy of
#' \code{Y} with zeros at missing locations for the objective evaluation.
#'
#' @param Y Numeric \code{n x p} count matrix. May contain \code{NA}.
#' @param X Numeric design matrix with \code{n*p} rows and \code{d} columns,
#'   aligned with \code{vec(Y)} (column‑wise vectorization).
#' @param q Integer, latent rank (dimension of the latent space).
#' @param params Optional list of initial parameters. If \code{NULL},
#'   \code{\link{Init_ZIP}} is used.
#' @param config_vem List of controls for the outer blockwise loop
#'   (variational EM‑like), with fields:
#'   \describe{
#'     \item{\code{maxiter}}{Maximum number of outer iterations (default \code{1e6}).}
#'     \item{\code{ftol}}{ELBO tolerance; stop if \code{|ELBO_{t}-ELBO_{t-1}| <= ftol}.}
#'     \item{\code{xtol}}{Parameter tolerance; stop if \code{max|theta_{t}-theta_{t-1}| <= xtol}.}
#'     \item{\code{tolS}}{List with \code{lower}, \code{upper} bounds for \code{S}.}
#'     \item{\code{tolxi}}{Tolerance used in the Jaakkola‑type \eqn{\xi} updates for the logistic bound.}
#'   }
#' @param config List of controls for the NLOpt backend (per‑block inner solve),
#'   e.g. \code{algorithm}, \code{maxeval}, \code{ftol_abs}, \code{xtol_abs},
#'   \code{ftol_rel}, \code{xtol_rel}, \code{trace}, \code{maxtime}, as well as
#'   \code{upper_bounds}, \code{lower_bounds} (filled here from \code{tolS} and dimensions).
#'
#' @return A list with components:
#' \describe{
#'   \item{\code{mStep}}{Model parameters: \code{gamma} (\code{d x 1}),
#'     \code{beta} (\code{d x 1}), loadings \code{C} (\code{p x q}).}
#'   \item{\code{eStep}}{Variational parameters: \code{M} (\code{n x q}),
#'     \code{S} (\code{n x q}), logistic bound parameters \code{xi} (\code{n x p}).}
#'   \item{\code{pred}}{List with predictors and expected counts:
#'     \code{mu} (\code{n x p}), \code{nu} (\code{n x p}), backend mean \code{A} (\code{n x p}),
#'     and \code{predicted} recomputed in R.}
#'   \item{\code{imputed}}{\code{n x p} matrix equal to \code{xi * A} at missing entries of \code{Y},
#'     and \code{Y} elsewhere.}
#'   \item{\code{iter}}{Number of outer iterations performed.}
#'   \item{\code{elboPath}}{Numeric vector of ELBO values across outer iterations.}
#'   \item{\code{elbo}}{Final ELBO value.}
#'   \item{\code{params.init}}{Parameters used for initialization.}
#'   \item{\code{monitoring}}{List with \code{status} (stopping reason) and \code{iterations}.}
#'   \item{\code{gradB, gradD, gradC, gradM, gradS}}{ELBO gradients w.r.t. each parameter block
#'     (from \code{\link{Elbo_grad}}) at the solution.}
#' }
#'
#' @details
#' A binary mask \code{R = 1_{observed}(Y)} is built; a working matrix \code{Y.na}
#' replaces missings by 0 for the objective. Box constraints for \code{S} are set
#' using \code{config_vem$tolS} and injected into \code{config$upper_bounds} and
#' \code{config$lower_bounds}. Each block update calls the backend
#' \code{nlopt_optimize_ZIP_Steps(data, params, config, tolxi, active_blocks)} with
#' a single active block at a time.
#'
#' \strong{Predicted mean:} in this implementation, the Poisson mean used for the
#' R‑side recomputation is
#' \deqn{ \exp\!\big( \mu + M C^\top + 0.5\, S \,(C\odot C)^\top \big), }
#' where \eqn{\mu = X B} reshaped into \code{n x p}. Note that other functions in
#' le package emploient \eqn{0.5\,(S\odot S)\,(C\odot C)^\top}. Vérifie que la
#' paramétrisation voulue est cohérente avec le backend.
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 50; p <- 15; d <- 3; q <- 2
#' Y <- matrix(rpois(n*p, 2), n, p); Y[sample(length(Y), 30)] <- NA
#' X <- cbind(1, rnorm(n*p), rnorm(n*p))  # (n*p) x d, vectorized design
#'
#' fit <- Miss.ZIPLNPCA_Steps(Y, X, q)
#' fit$elbo
#' plot(fit$elboPath, type = "l", xlab = "outer iter", ylab = "ELBO")
#' str(fit$mStep); str(fit$eStep)
#' }
#'
#' @seealso \code{\link{Miss.ZIPLNPCA}}, \code{\link{Miss.ZIPLNPCA.logS}},
#'   \code{\link{Init_ZIP}}, \code{\link{Elbo_grad}}
#' @import PLNmodels
#' @export




Miss.ZIPLNPCA_Steps <- function(Y, X, q, params = NULL, config_vem = NULL, config = NULL) {
  n <- nrow(Y)
  p <- ncol(Y)
  d <- ncol(X)
 
  
  if (is.null(config_vem)) {
    config_vem <- list(maxiter = 1e06, ftol = 1e-10, xtol = 1e-10,
                       tolS = list(lower = 0, upper = Inf), tolxi = 1e-04)
  }
  
  if (is.null(config)) {
    config <- list(algorithm = "MMA", backend = "nlopt", maxeval = 1e04,
                   ftol_abs = 1e-10, xtol_abs = 1e-10, maxtime = -1, trace = 1, ftol_rel = 1e-15, xtol_rel = 1e-15)
  }
  
  R <- ifelse(is.na(Y), 0, 1)
  Y.na <- ifelse(R == 0, 0, Y)
  
  data <- list(Y = Y.na, R = R, X = X)
  
  uBound <- c(rep(Inf, (2*d)+(p*q)+(n*q)), rep(config_vem$tolS$upper, n*q))
  lBound <- c(rep(-Inf, (2*d)+(p*q)+(n*q)), rep(config_vem$tolS$lower, n*q))
  config$upper_bounds <- uBound
  config$lower_bounds <- lBound
  
  if (is.null(params)) {
    params <- Init_ZIP(Y, X, q)
  }
  
  params.init <- params
  params_new <- unlist(params)
  params_old <- params_new + 1
  
  elbo_new <- Elbo_grad(data, params, config_vem$tolxi)$objective
  elbo_old <- elbo_new + 1
  iter <- 0
  elboPath <- c(elbo_new)
  status <- NULL
  
  while ((max(abs(params_new - params_old)) > config_vem$xtol &&
          abs(elbo_new - elbo_old) > config_vem$ftol) &&
         iter < config_vem$maxiter) {
    
    iter <- iter + 1
    cat("Iteration:", iter, "ELBO:", elbo_new, "Diff params:", 
        max(abs(params_new - params_old)), "Diff ELBO:", 
        abs(elbo_new - elbo_old), "\n")
    
    params_old <- params_new
    elbo_old <- elbo_new
    
    for (block in c("B", "D", "C", "M", "S")) {
      
      active_blocks <- list(B = FALSE, D = FALSE, C = FALSE, M = FALSE, S = FALSE)
      active_blocks[[block]] <- TRUE
      
      out_opt <- nlopt_optimize_ZIP_Steps(data, params, config, config_vem$tolxi, active_blocks)
      params$B <- out_opt$B
      params$D <- out_opt$D
      params$C <- out_opt$C
      params$M <- out_opt$M
      params$S <- out_opt$S
      
      grad <- Elbo_grad(data, params, config_vem$tolxi)
      cat(sprintf("gradB : %.4e gradD : %.4e gradC : %.4e gradM : %.4e gradS : %.4e\n",
                  mean(grad$gradB^2), mean(grad$gradD^2),
                  mean(grad$gradC^2), mean(grad$gradM^2),
                  mean(grad$gradS^2)))
    }
    
    params_new <- unlist(params)
    elbo_new <- out_opt$objective
    elboPath <- c(elboPath, elbo_new)
    
    if (max(abs(params_new - params_old)) <= config_vem$xtol) {
      status <- "xtol atteint"
      break
    }
    if (abs(elbo_new - elbo_old) <= config_vem$ftol) {
      status <- "ftol atteint"
      break
    }
  }
  
  if (is.null(status)) status <- "maxiter atteint"
  
  mu <- VectorToMatrix(X %*% out_opt$B, n, p)
  nu <- VectorToMatrix(X %*% out_opt$D, n, p)
  predicted <- exp(mu + out_opt$M %*% t(out_opt$C) + 
                     0.5 * out_opt$S %*% t(out_opt$C^2))
  
  imputed <- ifelse(is.na(Y), out_opt$xi*out_opt$A, Y)
  params <- list(B = out_opt$B, D = out_opt$D, C = out_opt$C, 
                 M = out_opt$M, S = out_opt$S)
  grad <- Elbo_grad(data, params, config_vem$tolxi)
  
  res <- list(
    mStep = list(gamma = out_opt$D, beta = out_opt$B, C = out_opt$C),
    eStep = list(M = out_opt$M, S = out_opt$S, xi = out_opt$xi),
    pred = list(A = out_opt$A, mu = mu, nu = nu, predicted = predicted),
    imputed = imputed,
    iter = iter,
    elboPath = elboPath,
    elbo = out_opt$objective,
    params.init = params.init,
    monitoring = list(status = status, iterations = iter),
    gradB = grad$gradB,
    gradD = grad$gradD,
    gradC = grad$gradC,
    gradM = grad$gradM, 
    gradS = grad$gradS
  )
  
  return(res)
}
