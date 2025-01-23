#' Init_ZIP_q0
#'
#' Parameters initialisation in the zero inflated case
#' @param Y count matrix
#' @param X covariates
#' @param q size of the latent space
#' @return A list with elements:
#'   \describe{
#'     \item{B}{Matrix of Poisson regression coefficients (d x p).}
#'     \item{D}{Matrix of logistic regression coefficients (d x p).}
#'   }
#' @export

Init_ZIP_q0 <- function(Y, X, q){
  
  n <- nrow(Y)
  p <- ncol(Y)
  vecY <- MatrixToVector(Y)

  fit <- lm(log(1 + vecY) ~ -1 + X, na.action = na.exclude)
  B <- as.matrix(fit$coefficients)

  
  U <- ifelse(Y == 0, 0, 1)
  vecU <- MatrixToVector(U)
  fit.logit <- glm(vecU ~ -1 + X, family = "binomial", na.action = na.exclude)
  D <- as.matrix(fit.logit$coefficients)

  
  return(list(B = B, D = D))
}

