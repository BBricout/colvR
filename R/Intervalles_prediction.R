#' Predictive simulations and intervals
#'
#' Draws parameter samples using the sandwich variance \code{V_theta}, simulates
#' predictive counts with zero-inflation and returns per-cell prediction
#' intervals. Two types of intervals are available: conditional (latent
#' variables drawn from their variational posterior, after one VE step per
#' draw) and marginal (latent variables drawn from their prior).
#'
#' @param Y An n x p count matrix (NAs allowed).
#' @param X An (n*p) x d design matrix (stacked by blocks of size n).
#' @param fit Estimator output list (from \code{Miss.ZIPLNPCA}).
#' @param MC Integer, number of Monte Carlo draws.
#' @param type \code{"conditional"} (default) or \code{"marginal"}.
#'   \describe{
#'     \item{conditional}{for each parameter draw, a VE step gives
#'       \eqn{M}, \eqn{S} and \eqn{\xi}; then
#'       \eqn{U \sim \mathcal{B}(\xi)} and
#'       \eqn{Z_i \sim \mathcal{N}(X\beta + M_i C^\top, C\,\mathrm{diag}(S_i)\,C^\top)}.}
#'     \item{marginal}{no VE step;
#'       \eqn{U \sim \mathcal{B}(\mathrm{logit}^{-1}(X\gamma))} and
#'       \eqn{Z_i \sim \mathcal{N}(X\beta, C C^\top)}.}
#'   }
#'   In both cases \eqn{U = 1} means presence and the prediction is
#'   \eqn{U \times \mathcal{P}(\exp(Z))}.
#' @param probs Numeric vector of length 2, the lower and upper quantile
#'   levels of the interval. Default \code{c(0, 0.9)}.
#' @param output Character vector, what to return besides the intervals.
#'   One or several of \code{"intervals"} (always returned), \code{"draws"},
#'   \code{"latent"}, \code{"params"}, or \code{"all"}.
#' @param cl Parallelisation of the VE steps (conditional type only), passed to
#'   \code{\link[pbapply]{pblapply}}. \code{NULL} (default): sequential.
#'   An integer: number of forked processes (not available on Windows).
#'   A cluster from \code{\link[parallel]{makeCluster}}: works on all platforms.
#'   All random draws are made in the main process, so results for a given
#'   seed do not depend on \code{cl}.
#'
#' @return A list with elements:
#' \itemize{
#'   \item \code{lower}, \code{upper}: n x p matrices of predictive quantiles
#'     at levels \code{probs}.
#'   \item \code{draws} (if requested): list with \code{count} (Poisson draws)
#'     and \code{presence} (Bernoulli draws \eqn{U}), arrays n x p x MC.
#'     The prediction is \code{count * presence}.
#'   \item \code{latent} (if requested): list with \code{Z}, array n x p x MC
#'     of Gaussian latent draws, and for the conditional type \code{M},
#'     \code{S} (arrays n x q x MC) and \code{xi} (array n x p x MC)
#'     from the VE steps.
#'   \item \code{params} (if requested): list with \code{gamma}, \code{beta}
#'     (matrices MC x d) and \code{C} (matrix MC x (p*q), column-major
#'     vectorisation of C).
#'   \item \code{settings}: list with \code{type}, \code{probs} and \code{MC}.
#' }
#'
#' @examples
#' \dontrun{
#' pi_cond <- Predictions(Y, X, fit, MC = 100)
#' pi_marg <- Predictions(Y, X, fit, MC = 100, type = "marginal",
#'                        probs = c(0.05, 0.95))
#'
#' # Parallel VE steps on a server
#' cl <- parallel::makeCluster(8)
#' res <- Predictions(Y, X, fit, MC = 1000, output = "all", cl = cl)
#' parallel::stopCluster(cl)
#' }
#'
#' @seealso \code{\link{V_theta}}, \code{\link{Miss.ZIPLNPCA_VE}}
#' @importFrom mvtnorm rmvnorm
#' @importFrom pbapply pblapply
#' @importFrom stats rbinom rpois plogis quantile
#' @export
Predictions <- function(Y, X, fit, MC,
                        type   = c("conditional", "marginal"),
                        probs  = c(0, 0.9),
                        output = "intervals",
                        cl     = NULL){
  
  ## Vérification des options
  type <- match.arg(type)
  output <- match.arg(output,
                      c("intervals", "draws", "latent", "params", "all"),
                      several.ok = TRUE)
  if ("all" %in% output) output <- c("intervals", "draws", "latent", "params")
  
  if (!is.numeric(probs) || length(probs) != 2 || anyNA(probs) ||
      any(probs < 0 | probs > 1) || probs[1] > probs[2])
    stop("'probs' doit contenir deux niveaux entre 0 et 1, le premier inférieur au second.")
  
  if (is.numeric(cl) && cl > 1 && .Platform$OS.type == "windows")
    stop("Sous Windows, 'cl' ne peut pas être un nombre de cœurs. ",
         "Utiliser un cluster : cl <- parallel::makeCluster(4).")
  
  ## Paramètres
  n <- nrow(Y) ; p <- ncol(Y) ; d <- ncol(X)
  B.hat <- fit$mStep$beta ; D.hat <- fit$mStep$gamma ; C.hat <- fit$mStep$C
  q <- ncol(C.hat)
  M.hat <- fit$eStep$M ; S.hat <- fit$eStep$S
  
  ## Variance sandwich
  Var <- as.matrix(V_theta(Y, X, fit))
  
  ## Tirages de paramètres
  theta.hat <- c(as.vector(D.hat), as.vector(B.hat), as.vector(C.hat))
  theta.sample <- lapply(1:MC, function(m)
    rmvnorm(1, theta.hat, Var))
  
  gamma.sample <- lapply(1:MC, function(m)
    as.matrix(theta.sample[[m]][1:d]))
  beta.sample <- lapply(1:MC, function(m)
    as.matrix(theta.sample[[m]][(d+1):(2*d)]))
  C.sample <- lapply(1:MC, function(m)
    VectorToMatrix(theta.sample[[m]][(2*d+1):(2*d +p*q)], p, q))
  
  mu.sample <- lapply(1:MC, function(m)
    VectorToMatrix(X%*%beta.sample[[m]], n, p))
  
  if (type == "conditional"){
    
    params.sample <- lapply(1:MC, function(m)
      list(B = matrix(beta.sample[[m]]), D = matrix(gamma.sample[[m]]), C = C.sample[[m]], M = M.hat, S = S.hat))
    
    ## Etapes VE (seule partie parallélisée, sans tirage aléatoire)
    config_VE <- list(algorithm = "MMA", backend = "nlopt", maxeval = 1000,
                      ftol_abs = 1e-8, xtol_abs = 1e-4, maxtime = -1, trace = 0,
                      ftol_rel = 1e-15, xtol_rel = 1e-15)
    
    VE.sample <- pblapply(1:MC, function(m)
      Miss.ZIPLNPCA_VE(Y, X, q, params = params.sample[[m]], config = config_VE),
      cl = cl)
    
    xi.VE <- lapply(1:MC, function(m)
      VE.sample[[m]]$eStep$xi)
    
    U <- lapply(1:MC, function(m)
      matrix(rbinom(n*p, prob = xi.VE[[m]], size = 1), nrow = n))
    
    mu <- lapply(1:MC, function(m)
      mu.sample[[m]] + VE.sample[[m]]$eStep$M %*% t(C.sample[[m]]))
    
    sigma <- lapply(1:MC, function(m){
      lapply(1:n, function(i)
        C.sample[[m]] %*% diag(VE.sample[[m]]$eStep$S[i,], nrow = q) %*% t(C.sample[[m]]))
    })
    
    Z <- lapply(1:MC, function(m){do.call(rbind, lapply(1:n, function(i){rmvnorm(1, mu[[m]][i,], sigma[[m]][[i]])}))})
    
  } else {
    
    nu <- lapply(1:MC, function(m)
      VectorToMatrix(X%*%gamma.sample[[m]], n, p))
    pi <- lapply(1:MC, function(m)
      plogis(nu[[m]]))
    
    U <- lapply(1:MC, function(m)
      matrix(rbinom(n*p, size = 1, prob = pi[[m]]), nrow = n, ncol = p))
    
    sigma <- lapply(1:MC, function(m)
      C.sample[[m]]%*%t(C.sample[[m]]))
    
    Z <- lapply(1:MC, function(m){do.call(rbind, lapply(1:n, function(i){rmvnorm(1, mu.sample[[m]][i,], sigma[[m]])}))})
    
  }
  
  ## Simulation des comptages
  Z_clipped <- lapply(Z, function(z) pmin(z, 700))
  
  Y.hat <- lapply(1:MC, function(m)
    matrix(rpois(n*p, lambda = exp(Z_clipped[[m]])), nrow = n))
  
  pred <- lapply(1:MC, function(m)
    Y.hat[[m]]*U[[m]])
  
  Y_sim_arr <- simplify2array(pred)
  
  ## Intervalles
  Y_pred_lower <- apply(Y_sim_arr, c(1, 2), function(x) quantile(x, probs[1], names = FALSE))
  Y_pred_upper <- apply(Y_sim_arr, c(1, 2), function(x) quantile(x, probs[2], names = FALSE))
  
  res <- list(lower = Y_pred_lower, upper = Y_pred_upper)
  
  ## Sorties optionnelles
  if ("draws" %in% output){
    res$draws <- list(count    = simplify2array(Y.hat),
                      presence = simplify2array(U))
  }
  
  if ("latent" %in% output){
    res$latent <- list(Z = simplify2array(Z))
    if (type == "conditional"){
      res$latent$M  <- simplify2array(lapply(VE.sample, function(ve) ve$eStep$M))
      res$latent$S  <- simplify2array(lapply(VE.sample, function(ve) ve$eStep$S))
      res$latent$xi <- simplify2array(xi.VE)
    }
  }
  
  if ("params" %in% output){
    theta.mat <- do.call(rbind, theta.sample)
    res$params <- list(gamma = theta.mat[, 1:d, drop = FALSE],
                       beta  = theta.mat[, (d+1):(2*d), drop = FALSE],
                       C     = theta.mat[, (2*d+1):(2*d + p*q), drop = FALSE])
  }
  
  res$settings <- list(type = type, probs = probs, MC = MC)
  
  return(res)
  
}