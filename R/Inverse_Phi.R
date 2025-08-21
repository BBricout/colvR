#' @keywords internal
#' @noRd

inv.HessPhi <- function(Y, X, fit){
  
  R <- ifelse(is.na(Y), 0, 1)
  Y.na <- ifelse(is.na(Y), 0, Y)
  B <- fit$mStep$beta ; D <- fit$mStep$gamma ; C <- fit$mStep$C
  M <- fit$eStep$M ; S <- fit$eStep$S ; A <- fit$pred$A  
  n <- nrow(Y) ; p <- ncol(Y) ; d <- ncol(X) ; q <- ncol(C)
  mu <- VectorToMatrix(X%*%B, n, p) ; nu <- VectorToMatrix(X%*%D, n, p)
  pi <- 1/(1 + exp(-nu)) ; xi <- fit$eStep$xi
  
  HessM <- grad2M(R, xi, A, C, n, q, p)
  HessS <- grad2S(R, xi, A, C, S, n, p)
  Hessxi <- grad2xi(xi, Y.na, R)
  MS <- gradMS(R, xi, A, C, n, p)
  Mxi <- gradMxi(R, Y.na, A, C, n, p, q)
  Sxi <- gradSxi(R, Y.na, A, C, n, p, q)
  
  
  pos <- lapply(1:n, function(i)
    which(Y.na[i,] >0))
  
  
  BlocA <- lapply(1:n, function(i)
    bdiag(HessM[[i]], HessS[[i]]))
  
  
  for (i in 1:n){BlocA[[i]][(q+1):(2*q), 1:q] <- MS[[i]]}
  
  for (i in 1:n){BlocA[[i]][1:q, (q+1):(2*q)] <- t(MS[[i]])}
  
  
  
  ## Fabrication de B
  
  
  
  BlocC <- lapply(1:n, function(i) 
    cbind(Mxi[[i]], Sxi[[i]]))
  
  BlocC <- lapply(1:n, function(i){
    if (length(pos[[i]]) != 0){
      matrix(BlocC[[i]][-pos[[i]],], ncol = 2*q)
    }
    else{
      matrix(BlocC[[i]], ncol = 2*q)
    }
  })
  
  
  
  ## Fabrication de C
  
  BlocB <- lapply(1:n, function(i)
    t(BlocC[[i]]))
  
  
  ## Fabrication de D
  
  BlocD <- lapply(1:n, function(i){
    if(length(pos[[i]]) != 0){
      diag(Hessxi[i,])[-pos[[i]],-pos[[i]]]
    }
    else{
      diag(Hessxi[i,])
    }
  }
  )
  
  inv.Hessxi <- inv.Grad2Xi(xi, Y, R)
  
  inv.BlocD <- lapply(1:n, function(i){
    if(length(pos[[i]]) != 0){
      diag(inv.Hessxi[i,])[-pos[[i]],-pos[[i]]]
    }
    else{
      diag(inv.Hessxi[i,])
    }
  }
  )
  
  
  
  ## Calcul de l'inverse
  
  Inverse <- vector("list", n)
  for (i in 1:n){
    if (length(pos[[i]]) != p){
      Bloc1 <- BlocA[[i]] - BlocB[[i]] %*% inv.BlocD[[i]] %*% BlocC[[i]]
      inv.Bloc1 <- solve(Bloc1)
      Bloc2 <- -inv.Bloc1 %*% BlocB[[i]] %*% inv.BlocD[[i]]
      Bloc3 <- -inv.BlocD[[i]] %*% BlocC[[i]] %*% inv.Bloc1
      Bloc4 <- inv.BlocD[[i]] + inv.BlocD[[i]] %*% BlocC[[i]] %*% inv.Bloc1 %*% BlocB[[i]] %*% inv.BlocD[[i]]
      Top <- cbind(inv.Bloc1, Bloc2)
      Bottom <- cbind(Bloc3, Bloc4)
      Inverse[[i]] <- rbind(Top, Bottom)
    }
    else{
      Inverse[[i]] <- solve(BlocA[[i]])
    }
  }
  
  
  
  return(Inverse)
}
