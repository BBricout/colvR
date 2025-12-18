## Dérivées secondes

#' @keywords internal
#' @noRd

grad2gamma <- function(xi, pi, X, n, p){
  indices <- lapply(1:n, function(i) 
    sapply(1:p-1, function(k) k * n + i))
  grad2 <- vector("list", n)
  for (i in 1:n){
    grad2[[i]] <- 0
    for (j in 1:p){
      grad2[[i]] <- grad2[[i]] + pi[i,j]*(pi[i,j] - 1) * (X[indices[[i]][j],])%*%t(X[indices[[i]][j],])
    }
  }
  return(grad2)
}

#' @keywords internal
#' @noRd

grad2beta <- function(R, xi, A, X, n, p){
  indices <- lapply(1:n, function(i) 
    sapply(1:p-1, function(k) k * n + i))
  grad2 <- vector("list", n)
  for (i in 1:n){
    grad2[[i]] <- 0
    for (j in 1:p){
      grad2[[i]] <- grad2[[i]] +(R[i,j]*xi[i,j]*A[i,j]) * (X[indices[[i]][j],])%*%t(X[indices[[i]][j],])
    }
    grad2[[i]] <- -grad2[[i]]
  }
  return(grad2)
}

#' @keywords internal
#' @noRd

grad2C <- function(R, xi, A, M, C, S, n, p){
  grad2 <- vector("list", p)
  for (j in 1:p){
    grad2[[j]] <- lapply(1:n, function(i)
      - R[i,j]*xi[i,j]*A[i,j] * (M[i,]%*%t(M[i,]) + (C[j,]*S[i,])%*%t(M[i,]) + M[i,] %*% t(C[j,] * S[i,]) + (C[j,]*S[i,])%*%t(C[j,]*S[i,]) + diag(S[i,])))
  }
  return(grad2)
}

#' @keywords internal
#' @noRd

grad2M <- function(R, xi, A, C, n, q, p){
  grad2 <- vector("list", n)
  for (i in 1:n){
    grad2[[i]] <- 0
    for (j in 1:p){
      grad2[[i]] <- grad2[[i]] + R[i,j] * xi[i,j] * A[i,j] * C[j,] %*% t(C[j,])
    }
    grad2[[i]] <- - grad2[[i]] - diag(1, q, q)
  }
  return(grad2)
}

#' @keywords internal
#' @noRd

grad2S <- function(R, xi, A, C, S, n, p){
  grad2 <- vector("list", n)
  for (i in 1:n){
    grad2[[i]] <- 0
    for (j in 1:p){
      grad2[[i]] <- grad2[[i]] + R[i,j] * xi[i,j] * A[i,j] * (C[j,]*C[j,])%*%t(C[j,]*C[j,])
    }
    grad2[[i]] <- -0.5 *((diag(S[i,]**(-2))) + 0.5 * grad2[[i]])
  }
  return(grad2)
}

#' @keywords internal
#' @noRd

grad2xi <- function(xi, Y, R){
  grad <- ifelse((Y == 0 & R == 1) | (R == 0), -((xi*(1 - xi)))**(-1), 0)
  return(grad)
}

#' @keywords internal
#' @noRd

inv.Grad2Xi <- function(xi, Y, R){
  grad <- ifelse((Y == 0 & R == 1) | (R == 0), (-(xi*(1 - xi))), 0)
  return(grad)
}


## Dérivées croisées

#' @keywords internal
#' @noRd

gradBC <- function(R, X, xi, A, M, C, S, n, p){
  indices <- lapply(1:n, function(i) 
    sapply(1:p-1, function(k) k * n + i))
  grad <- vector("list", p)
  for (j in 1:p){
    grad[[j]] <- lapply(1:n, function(i)
      #  -R[i,j] * xi[i,j] * A[i,j] * (preptoX(M[i,], p) + preptoX(vec = (C[j,] * S[i,]), p = p)) %*% X[indices[[i]],])
      -R[i,j] * xi[i,j] * A[i,j] * (M[i,] + C[j,] * S[i,])%*% t(X[indices[[i]][j], ]))
  }
  return(grad)
}

#' @keywords internal
#' @noRd

gradMS <- function(R, xi, A, C, n, p){
  grad2 <- vector("list", n)
  for (i in 1:n){
    grad2[[i]] <- 0
    for (j in 1:p){
      grad2[[i]] <- grad2[[i]] + R[i,j] * xi[i,j] * A[i,j] * (C[j,]*C[j,])%*%t(C[j,])
    }
    grad2[[i]] <- -0.5 *grad2[[i]]
  }
  return(grad2)
}

#' @keywords internal
#' @noRd

gradMxi <- function(R, Y, A, C, n, p, q){
  grad2 <- vector("list", n)
  
  for (i in 1:n){
    grad <- matrix(ncol = q, nrow = p)
    for (j in 1:p){
      if ((Y[i,j] == 0 & R[i,j] == 1) | (R[i,j] == 0)){grad[j,] <- R[i,j] * (Y[i,j] - A[i,j]) * C[j,]}
      else {grad[j,] <- 0}
    }
    grad2[[i]] <- grad
  }
  
  return(grad2)
}

#' @keywords internal
#' @noRd

gradSxi <- function(R, Y, A, C, n, p, q){
  grad2 <- vector("list", n)
  
  for (i in 1:n){
    grad <- matrix(ncol = q, nrow = p)
    for (j in 1:p){
      if((Y[i,j] == 0 & R[i,j] == 1)| (R[i,j] == 0)){grad[j,] <- -0.5 * R[i,j] * A[i,j] * (C[j,] * C[j,])}
      else {grad[j,] <- 0}
    }
    grad2[[i]] <- grad
  }
  
  return(grad2)
}

#' @keywords internal
#' @noRd

gradBM <- function(R, xi, A, C, X, n, p){
  indices <- lapply(1:n, function(i) 
    sapply(1:p-1, function(k) k * n + i))
  grad2 <- vector("list", n)
  for (i in 1:n){
    grad2[[i]] <- 0
    for (j in 1:p){
      grad2[[i]] <- grad2[[i]] + R[i,j] * xi[i,j] * A[i,j] * C[j,] %*% t(X[indices[[i]][j],])
    }
    grad2[[i]] <- -grad2[[i]]
  }
  return(grad2)
}


#' @keywords internal
#' @noRd

gradBS <- function(R, xi, A, C, X, n, p){
  indices <- lapply(1:n, function(i) 
    sapply(1:p-1, function(k) k * n + i))
  grad2 <- vector("list", n)
  for (i in 1:n){
    grad2[[i]] <- 0
    for (j in 1:p){
      grad2[[i]] <- grad2[[i]] + R[i,j] * xi[i,j] * A[i,j] * (C[j,] * C[j,]) %*% t(X[indices[[i]][j],])
    }
    grad2[[i]] <- -0.5 * grad2[[i]]
  }
  return(grad2)
}

#' @keywords internal
#' @noRd

gradMC <- function(R, xi, A, Y, C, M, S, n, p, q){
  grad <- vector("list", p)
  for (j in 1:p){
    grad[[j]] <- lapply(1:n, function(i)
      R[i,j]*xi[i,j]*(Y[i,j]*diag(1,q,q) - A[i,j]*((M[i,] + S[i,]*C[j,])%*% t(C[j,]) + diag(1, q, q))))
  }
  return(grad)
}

#' @keywords internal
#' @noRd

gradSC <- function(R, xi, A, C, M, S, n, p){
  grad <- vector("list", p)
  
  for (j in 1:p){
    grad[[j]] <- lapply(1:n, function(i)
      R[i,j] * xi[i,j] * A[i,j] * (-0.5*(C[j,]*C[j,])%*%(t(M[i,]) + t(C[j,] * S[i,])) - diag(C[j,])))
  }
  
  return(grad)
}

#' @keywords internal
#' @noRd

gradGammaXi <- function(Y, X, n, p, d){
  indices <- lapply(1:n, function(i) 
    sapply(1:p-1, function(k) k * n + i))
  grad <- lapply(1:n, function(i)
    X[indices[[i]],])
  
  for (i in 1:n){
    for (j in 1:p){
      if (Y[i,j] > 0){
        grad[[i]][j,] <- rep(0, d) 
      }
    }
  }
  return(grad)
}

#' @keywords internal
#' @noRd


gradBxi <- function(R, Y, A, X, n, p, d){
  indices <- lapply(1:n, function(i) 
    sapply(1:p-1, function(k) k * n + i))
  grad <- lapply(1:n, function(i)
    diag(R[i,]*(Y[i,] - A[i,])) %*% X[indices[[i]],])
  for (i in 1:n){
    for (j in 1:p){
      if (Y[i,j] > 0){
        grad[[i]][j,] <- rep(0, d) 
      }
    }
  }
  
  return(grad)
}

#' @keywords internal
#' @noRd

gradCxi <- function(R, Y, A, M, C, S, n, p, q){
  grad <- vector("list", p)
  
  for (j in 1:p){
    grad[[j]] <- lapply(1:n, function(i)
      R[i,j] * ((Y[i,j]-A[i,j])*M[i,] - A[i,j]*(C[j,]*S[i,])))
  }
  
  for (i in 1:n){
    for (j in 1:p){
      if (Y[i,j] > 0){
        grad[[j]][[i]] <- rep(0, q)
      }
    }
  }
  
  return(grad)
}

























