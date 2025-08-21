#' @keywords internal
#' @noRd

GradGamma <- function(X, xi, pi, n, p){
  indices <- lapply(1:n, function(i) 
    sapply(1:p-1, function(k) k * n + i))
  grad <- vector("list", n)
  for (i in 1:n){
    grad[[i]] <- 0
    for (j in 1:p){
      grad[[i]] <- grad[[i]] + (xi[i,j] - pi[i,j])*X[indices[[i]][j],]
    }
  }
  return(grad)
}

#' @keywords internal
#' @noRd

GradBeta <- function(Y, R, X, A, xi, n, p){
  indices <- lapply(1:n, function(i) 
    sapply(1:p-1, function(k) k * n + i))
  grad <- vector("list", n)
  for (i in 1:n){
    grad[[i]] <- 0
    for (j in 1:p){
      grad[[i]] <- grad[[i]] + R[i,j] * xi[i,j] *(Y[i,j] - A[i,j]) * X[indices[[i]][j],]
    }
  }
  return(grad)
}

#' @keywords internal
#' @noRd


GradC <- function(Y, R, A, C, xi, M, S, n, p){
  grad <- vector("list", p)
  for (j in 1:p){
    grad[[j]] <- lapply(1:n, function(i)
      R[i,j] * xi[i,j]* ((Y[i,j] - A[i,j])*M[i,] - A[i,j]*(C[j,]*S[i,])))
  }
  return(grad)
}

#' @keywords internal
#' @noRd

GradTheta <- function(Y, X, fit){
  
  R <- ifelse(is.na(Y), 0, 1)
  Y.na <- ifelse(is.na(Y), 0, Y)
  B <- fit$mStep$beta ; D <- fit$mStep$gamma ; C <- fit$mStep$C
  M <- fit$eStep$M ; S <- fit$eStep$S ; A <- fit$pred$A 
  n <- nrow(Y) ; p <- ncol(Y) ; d <- ncol(X) ; q <- ncol(C)
  mu <- VectorToMatrix(X%*%B, n, p) ; nu <- VectorToMatrix(X%*%D, n, p)
  pi <- 1/(1 + exp(-nu)) ; xi <- fit$eStep$xi
  
  hessgamma <- GradGamma(X, xi, pi, n, p)
  hessbeta <- GradBeta(Y.na, R, X, A, xi, n, p)
  hessC <- GradC(Y.na, R, A, C, xi, M, S, n, p)
  
  
  
  DiagGradTheta <- lapply(1:n, function(i) {
    do.call(c, list(
      list(hessgamma[[i]]), 
      list(hessbeta[[i]]), 
      lapply(1:p, function(j) hessC[[j]][[i]])
    ))
  })
  
  GradTheta <- lapply(1:n, function(i)
    diag(unlist(DiagGradTheta[[i]])))
  
  
  return(GradTheta)
  
}





