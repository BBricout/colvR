#' Simulate zero-inflated Poisson log-normal PCA data
#'
#' Generates simulated observations \code{Y} under a ZIP-LN-PCA structure from
#' parameters in \code{theta} and dimensions in \code{dim}.
#'
#' @param X An \eqn{(n \times p) \cdot d} design matrix stacked so that
#'   \code{VectorToMatrix(X \%*\% B, n, p)} produces the mean matrix \code{mu}.
#' @param theta List of model parameters with components:
#'   \itemize{
#'     \item \code{B}: coefficient matrix (d x 1 or compatible)
#'     \item \code{D}: zero-inflation coefficient matrix (d x 1 or compatible)
#'     \item \code{C}: loading matrix (p x q)
#'   }
#' @param dim List of dimensions with entries \code{n}, \code{p}, \code{d}, \code{q}.
#'
#' @return A list with elements:
#'   \itemize{
#'     \item \code{Y}: simulated counts (n x p)
#'     \item \code{W}: latent scores (n x q)
#'     \item \code{U}: zero-inflation indicators (n x p)
#'     \item \code{Z}: Poisson draws before zero-inflation (n x p)
#'     \item \code{Lambda}: Poisson intensities (n x p)
#'     \item \code{mu}: mean matrix (n x p)
#'   }
#' @details Relies on \code{VectorToMatrix()} being available in the package.
#' @examples
#' set.seed(1)
#' n <- 30; p <- 5; d <- 2; q <- 2
#' X <- cbind(1, rnorm(n * p))                       # (n*p) x d design
#' theta <- list(B = matrix(c(1, 0.5)), D = matrix(c(1, -0.5)),
#'               C = matrix(rnorm(p * q, sd = 0.5), p, q))
#' sim <- Simul(X, theta, list(n = n, p = p, d = d, q = q))
#' str(sim)
#' mean(sim$Y == 0)                                  # proportion of zeros
#' @export
Simul <- function(X,theta, dim){
  q <- dim$q
  p <- dim$p
  n <- dim$n
  d <- dim$d
  
  B <- theta$B
  D <- theta$D
  C <- theta$C
  
  mu <- VectorToMatrix(X%*%B, n, p)
  nu <- VectorToMatrix(X%*%D, n, p)
  
  
  Prob <- plogis(nu)
  U <- matrix(rbinom(n*p, prob = Prob, size = 1), nrow = n)
  
  W <- matrix(rnorm(n*q), nrow = n)
  CW <- W %*% t(C)
  
  Lambda <- exp(mu + CW)
  Z <- matrix(rpois(n*p, lambda = Lambda), nrow = n)
  
  Y <- U*Z
  
  return(list(Y = Y, W = W, U = U,Z=Z, Lambda = Lambda, mu = mu))
  
}
