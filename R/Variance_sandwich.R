# Imports de fonctions externes (sans attacher de package)
#' @keywords internal
#' @importFrom Matrix bdiag
NULL


#' @keywords internal
#' @noRd
HessTheta <- function(Y, X, fit){
  R <- ifelse(is.na(Y), 0, 1)
  Y.na <- ifelse(is.na(Y), 0, Y)
  B <- fit$mStep$beta ; D <- fit$mStep$gamma ; C <- fit$mStep$C
  M <- fit$eStep$M ; S <- fit$eStep$S ; A <- fit$pred$A  
  n <- nrow(Y) ; p <- ncol(Y) ; d <- ncol(X) ; q <- ncol(C)
  mu <- VectorToMatrix(X%*%B, n, p) ; nu <- VectorToMatrix(X%*%D, n, p)
  pi <- 1/(1 + exp(-nu)) ; xi <- fit$eStep$xi
  
  hessgamma <- grad2gamma(xi, pi, X, n, p)
  hessbeta  <- grad2beta(R, xi, A, X, n, p)
  hessC     <- grad2C(R, xi, A, M, C, S, n, p)
  hessBC    <- gradBC(R = R, X = X, xi = xi, A = A, M = M, C = C, S = S, n = n, p = p)
  
  DiagGrad2Theta <- lapply(1:n, function(i) {
    do.call(c, list(
      list(hessgamma[[i]]), 
      list(hessbeta[[i]]), 
      lapply(1:p, function(j) hessC[[j]][[i]])
    ))
  })
  
  Grad2Theta <- lapply(1:n, function(i)
    bdiag(DiagGrad2Theta[[i]]))
  
  colBC <- lapply(1:n, function(j) {
    do.call(rbind, lapply(hessBC, `[[`, j))
  })
  
  for (i in 1:n){
    Grad2Theta[[i]][(2*d+1):(2*d + p*q), (d+1):(2*d)] <- colBC[[i]]
  }
  for (i in 1:n){
    Grad2Theta[[i]][(d+1):(2*d), (2*d+1):(2*d + p*q)] <- t(colBC[[i]])
  }
  return(Grad2Theta)
}

#' @keywords internal
#' @noRd
HessPhi <- function(Y, X, fit){
  R <- ifelse(is.na(Y), 0, 1)
  Y.na <- ifelse(is.na(Y), 0, Y)
  B <- fit$mStep$beta ; D <- fit$mStep$gamma ; C <- fit$mStep$C
  M <- fit$eStep$M ; S <- fit$eStep$S ; A <- fit$pred$A  
  n <- nrow(Y) ; p <- ncol(Y) ; d <- ncol(X) ; q <- ncol(C)
  mu <- VectorToMatrix(X%*%B, n, p) ; nu <- VectorToMatrix(X%*%D, n, p)
  pi <- 1/(1 + exp(-nu)) ; xi <- fit$eStep$xi
  
  HessM <- grad2M(R, xi, A, C, n, q)
  HessS <- grad2S(R, xi, A, C, S, n)
  Hessxi <- grad2xi(xi, Y.na, R)
  MS  <- gradMS(R, xi, A, C, n)
  Mxi <- gradMxi(R, Y.na, A, C, n, p, q)
  Sxi <- gradSxi(R, Y.na, A, C, n, p, q)
  
  DiagGrad2Phi <- lapply(1:n, function(i)
    list(HessM[[i]], HessS[[i]], diag(Hessxi[i,])))
  
  Grad2Phi <- lapply(1:n, function(i)
    bdiag(DiagGrad2Phi[[i]]))
  
  for (i in 1:n){ Grad2Phi[[i]][(q+1):(2*q), 1:q] <- MS[[i]] }
  for (i in 1:n){ Grad2Phi[[i]][1:q, (q+1):(2*q)] <- t(MS[[i]]) }
  for (i in 1:n){ Grad2Phi[[i]][(2*q + 1):(2*q + p), 1:q] <- Mxi[[i]] }
  for (i in 1:n){ Grad2Phi[[i]][1:q, (2*q + 1):(2*q + p)] <- t(Mxi[[i]]) }
  for (i in 1:n){ Grad2Phi[[i]][(2*q + 1):(2*q + p), (q+1):(2*q)] <- Sxi[[i]] }
  for (i in 1:n){ Grad2Phi[[i]][(q+1):(2*q), (2*q + 1):(2*q + p)] <- t(Sxi[[i]]) }
  
  return(Grad2Phi)
}

#' @keywords internal
#' @noRd
HessPhiTheta <- function(Y, X, fit){
  R <- ifelse(is.na(Y), 0, 1)
  Y.na <- ifelse(is.na(Y), 0, Y)
  B <- fit$mStep$beta ; D <- fit$mStep$gamma ; C <- fit$mStep$C
  M <- fit$eStep$M ; S <- fit$eStep$S ; A <- fit$pred$A  
  n <- nrow(Y) ; p <- ncol(Y) ; d <- ncol(X) ; q <- ncol(C)
  mu <- VectorToMatrix(X%*%B, n, p) ; nu <- VectorToMatrix(X%*%D, n, p)
  pi <- 1/(1 + exp(-nu)) ; xi <- fit$eStep$xi
  
  BM <- gradBM(R, xi, A, C, X, n, p)
  BS <- gradBS(R, xi, A, C, X, n, p)
  CM <- gradMC(R, xi, A, Y.na, C, M, S, n, p, q)
  CS <- gradSC(R, xi, A, C, M, S, n, p)
  GammaXi <- gradGammaXi(Y.na, X, n, p, d)
  BXi <- gradBxi(R, Y.na, A, X, n, p, d)
  Cxi <- gradCxi(R, Y.na, A, M, C, S, n, p, q)
  
  GradThetaPhi <- lapply(1:n, function(i)
    matrix(0, nrow = 2*q + p, ncol = 2*d + p*q))
  
  rowMC <- lapply(1:n, function(j) {
    do.call(cbind, lapply(CM, `[[`, j))
  })
  rowSC <- lapply(1:n, function(j) {
    do.call(cbind, lapply(CS, `[[`, j))
  })
  rowxiC <- lapply(1:n, function(j) {
    do.call(cbind, lapply(Cxi, `[[`, j))
  })
  
  for (i in 1:n){
    GradThetaPhi[[i]][(2*q + 1):(2*q + p), 1:d] <- GammaXi[[i]]
    GradThetaPhi[[i]][1:q , (d+1): (2*d)] <- BM[[i]]
    GradThetaPhi[[i]][(q+1): (2*q), (d+1):(2*d)] <- BS[[i]]
    GradThetaPhi[[i]][(2*q + 1) : (2*q + p), (d+1):(2*d)] <- BXi[[i]]
    GradThetaPhi[[i]][1:q, (2*d + 1): (2*d + p*q)] <- rowMC[[i]]
    GradThetaPhi[[i]][(q+1):(2*q), (2*d + 1): (2*d + p*q)] <- rowSC[[i]]
    GradThetaPhi[[i]][(2*q+1):(2*q+p), (2*d + 1): (2*d + p*q)] <- rowxiC[[i]]
  }
  
  HThetaPhi <- lapply(1:n, function(i)
    t(GradThetaPhi[[i]]))
  return(HThetaPhi)
}

#' @keywords internal
#' @noRd
C_theta <- function(Y, X, fit){
  n <- nrow(Y) ; p <- ncol(Y) ; q <- ncol(fit$mStep$C)
  Y.na <- ifelse(is.na(Y), 0, Y)
  
  pos <- lapply(1:n, function(i)
    which(Y.na[i,] >0))
  
  HTheta <- HessTheta(Y, X, fit)
  HPhiTheta <- HessPhiTheta(Y, X, fit) 
  
  HPhiTheta.fullRank <- vector("list", n)
  for (i in 1:n){
    if (length(pos[[i]]) != 0){
      HPhiTheta.fullRank[[i]] <- HPhiTheta[[i]][, - (2*q + pos[[i]])]
    } else {
      HPhiTheta.fullRank[[i]] <- HPhiTheta[[i]]
    }
  }
  
  inv.HPhi <- inv.HessPhi(Y, X, fit)
  
  Hess <- lapply(1:n, function(i)
    HTheta[[i]] - HPhiTheta.fullRank[[i]] %*% inv.HPhi[[i]] %*% t(HPhiTheta.fullRank[[i]]))
  
  CTheta <- (1/n) * Reduce("+", Hess)
  return(CTheta)
}

#' @keywords internal
#' @noRd
D_theta <- function(Y, X, fit){
  R <- ifelse(is.na(Y), 0, 1)
  Y.na <- ifelse(is.na(Y), 0, Y)
  B <- fit$mStep$beta ; D <- fit$mStep$gamma ; C <- fit$mStep$C
  M <- fit$eStep$M ; S <- fit$eStep$S ; A <- fit$pred$A
  n <- nrow(Y) ; p <- ncol(Y) ; d <- ncol(X) ; q <- ncol(C)
  mu <- VectorToMatrix(X%*%B, n, p) ; nu <- VectorToMatrix(X%*%D, n, p)
  pi <- 1/(1 + exp(-nu)) ; xi <- fit$eStep$xi
  
  grad <- GradTheta(Y, X, fit)
  Di <- lapply(1:n, function(i)
    diag(grad[[i]]) %*% t(diag(grad[[i]])))
  D <- (1/n) * Reduce("+", Di)
  return(D)
}

#' Variance–covariance of theta
#'
#' Computes \eqn{V(\hat\theta)} via the sandwich formula.
#'
#' @importFrom MASS ginv
#' @param Y An n x p matrix (counts).
#' @param X An (n*p) x d matrix (design stacked in blocks of size n).
#' @param fit Output list from your estimator.
#' @return Symmetrized variance–covariance matrix.
#' @export

V_theta <- function(Y, X, fit, ginv_tol = 1e-12, verbose = TRUE) {
  n <- nrow(Y)
  
  # Construire Ctheta (et la forcer symétrique numériquement)
  Ctheta <- C_theta(Y, X, fit)
  Ctheta <- as.matrix(Ctheta)
  Ctheta <- 0.5 * (Ctheta + t(Ctheta))
  
  # Dtheta
  Dtheta <- D_theta(Y, X, fit)
  
  # Inversion "safe" : try solve(), sinon pseudo-inverse
  Cinv <- tryCatch(solve(Ctheta),
                   error = function(e) NA)
  
  # Si solve() échoue ou donne des non-finis, on utilise ginv()
  if (is.atomic(Cinv) && length(Cinv) == 1 && is.na(Cinv) ||
      any(!is.finite(Cinv))) {
    if (verbose) message("solve() a échoué ou est instable → utilisation de MASS::ginv().")
    if (!requireNamespace("MASS", quietly = TRUE)) {
      stop("solve() a échoué et le package 'MASS' n'est pas disponible pour ginv().")
    }
    Cinv <- MASS::ginv(Ctheta, tol = ginv_tol)
  }
  
  # Sandwich
  Vt <- Cinv %*% Dtheta %*% Cinv
  
  # Mise à l'échelle et symétrisation finale
  var <- (1 / n) * Vt
  var_sym <- 0.5 * (var + t(var))
  return(var_sym)
}






