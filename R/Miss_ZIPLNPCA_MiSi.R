#' Miss.ZIPLNPCA_MiSi
#'
#' Estimation of the parameters and the missing data
#' @param Y count matrix
#' @param X covariates
#' @param q size of the latent space
#' @param params Initial parameters
#' @param config_vem configuration of the optimizer
#' @param config configuration for the steps 
#' @return A list of the estimated parameters
#' @import PLNmodels
#' @export


Miss.ZIPLNPCA_MiSi <- function(Y, # Matrice de comptage 
                                 X, # Covariables
                                 q, # Dimension de l'espace latent
                                 params = NULL, # Paramètres fourni en entrée
                                 config_vem = NULL, # (maxiter, tolS, tolxi, ftol, xtol) 
                                 config = NULL){ # Configuration pour les étapes
  ## Dimensions
  n <- nrow(Y)
  p <- ncol(Y)
  d <- ncol(X)
  
  ## Configuration
  
  if(is.null(config_vem)){
    config_vem <- list(maxiter = 1e06, ftol = 1e-10, xtol = 1e-10, 
                       tolS = list(lower = 0, upper = Inf), tolxi = 1e-04)
  }
  
  if (is.null(config)){
    config <- list(algorithm = "MMA", backend = "nlopt", maxeval = 1e06,
                   ftol_abs = 1e-8, xtol_abs = 1e-8, maxtime = -1, trace = 1, ftol_rel = 1e-15, xtol_rel = 1e-15)
  }
  
  
  ftol <- config_vem$ftol
  xtol <- config_vem$xtol
  tolS <- config_vem$tolS
  tolxi <- config_vem$tolxi
  maxiter <- config_vem$maxiter
  
  ## Preparation 
  
  R <- ifelse(is.na(Y), 0, 1) # Masque qui met des 0 à la place des données manquantes
  
  Y.na <- ifelse(R == 0, 0, Y)
  
  data <- list(Y = Y.na,
               R = R,
               X = X)
  
  uBound <- c(rep(Inf, q), rep(tolS$upper, q))
  lBound <- c(rep(-Inf, q), rep(tolS$lower, q))
  configMiSi <- config
  configMiSi$lower_bounds <- lBound
  configMiSi$upper_bounds <- uBound
  
  ## Initialisation
  
  if(is.null(params)){
    params <- Init_ZIP(Y, X, q)
  }
  
  
  params.init <- params
  
  params_new <- unlist(params)
  params_old <- params_new + rep(1, length(params_new))
  
  elbo_new <- Elbo_grad(data, params, tolxi)$objective
  elbo_old <- elbo_new + 1
  
  iter <- 0
  
  elboPath <- c(elbo_new)
  
  status <- NULL
  
  S <- params.init$S ; M <- params.init$M
  
  ## VEM
  
  
  while((max(abs(params_new - params_old)) > xtol && abs(elbo_new - elbo_old) > ftol) && iter < maxiter){
    
    iter <- iter +1
    
    cat("Iteration:", iter, "ELBO:", elbo_new, "Diff params:", max(abs(params_new - params_old)), "Diff ELBO:", abs(elbo_new - elbo_old), "\n")
    
    params_old <- params_new
    elbo_old <- elbo_new
    
    OutM_BD <- nlopt_optimize_ZIP_M_BetaGamma(data, params, config, tolxi)
    params <- list(B = OutM_BD$B, D = OutM_BD$D, C = OutM_BD$C,
                   M = OutM_BD$M, S = OutM_BD$S)
    
    grad <- Elbo_grad(data, params, tolxi = 1e-04)
    
    cat("gradB : ", mean((grad$gradB**2)), "gradD : ", mean((grad$gradD**2)), 
        "gradC : ", mean((grad$gradC**2)), "gradM : ", mean((grad$gradM**2)), 
        "gradS : ", mean((grad$gradS**2)), "\n")
    
    OutM_C <- nlopt_optimize_ZIP_M_C(data, params, config, tolxi)
    params <- list(B = OutM_C$B, D = OutM_C$D, C = OutM_C$C,
                   M = OutM_C$M, S = OutM_C$S)
    
    grad <- Elbo_grad(data, params, tolxi = 1e-04)
    
    cat("gradB : ", mean((grad$gradB**2)), "gradD : ", mean((grad$gradD**2)), 
        "gradC : ", mean((grad$gradC**2)), "gradM : ", mean((grad$gradM**2)), 
        "gradS : ", mean((grad$gradS**2)), "\n")
    
    for (i in 0:(n-1)){
      outMi <- nlopt_optimize_ZIP_VE_Mi(data, params, configMiSi, tolxi, i)
      params$M[i,] <- t(outMi$Mi)
    }
    
    grad <- Elbo_grad(data, params, tolxi = 1e-04)
    
    cat("gradB : ", mean((grad$gradB**2)), "gradD : ", mean((grad$gradD**2)), 
        "gradC : ", mean((grad$gradC**2)), "gradM : ", mean((grad$gradM**2)), 
        "gradS : ", mean((grad$gradS**2)), "\n")
    
    for (i in 0:(n-1)){
      outSi <- nlopt_optimize_ZIP_VE_Si(data, params, configMiSi, tolxi, i)
      params$S[i,] <- t(outSi$Si)
    }
    
    grad <- Elbo_grad(data, params, tolxi = 1e-04)
    
    cat("gradB : ", mean((grad$gradB**2)), "gradD : ", mean((grad$gradD**2)), 
        "gradC : ", mean((grad$gradC**2)), "gradM : ", mean((grad$gradM**2)), 
        "gradS : ", mean((grad$gradS**2)), "\n")
    
    outVE <- list(
      B = params$B,
      D = params$D,
      C = params$C,
      M = params$M,
      S = params$S,
      A = outSi$A,       # facultatif
      xi = outSi$xi,     # facultatif
      elbo1 = grad$elbo1,
      elbo2 = grad$elbo2,
      elbo3 = grad$elbo3,
      elbo4 = grad$elbo4,
      elbo5 = grad$elbo5,
      objective = grad$objective
    )
    params_new <- unlist(params)
    elbo_new <- outVE$objective
    
    elboPath <- append(elboPath, elbo_new)
    
    if (iter %% 1000 == 0) {
      saveRDS(list(
        params = params, 
        elbo = elbo_new, 
        elboPath = elboPath, 
        iteration = iter), 
        file = paste0("iteration_", iter, "_results.rds"))
    }
    
    # plot(elboPath, type = "l", col = "blue", lwd = 2, xlab = "Iteration", ylab = "ELBO", main = "Convergence de ELBO", ylim = quantile(elboPath, probs = c(0.1, 1)))
    # Sys.sleep(0.1)  # Pause courte pour afficher le graphique
    
    # Vérification des conditions d'arrêt
    if (max(abs(params_new - params_old)) <= xtol) {
      status <- "xtol atteint"
      break
    }
    
    if (abs(elbo_new - elbo_old) <= ftol) {
      status <- "ftol atteint"
      break
    }
    
  }
  
  if (is.null(status)) {
    status <- "maxiter atteint"
    
  }
  
  ## Résultats
  out <- outVE
  
  mu <- VectorToMatrix(X%*%out$B, n, p)
  nu <- VectorToMatrix(X%*%out$D, n, p)
  
  mStep <- list(gamma = out$D, beta = out$B, C = out$C)
  eStep <- list(M = out$M, S = out$S,  xi = out$xi)
  
  B.hat <- mStep$beta
  D.hat <- mStep$gamma
  C.hat <- mStep$C
  M.hat <- eStep$M
  S.hat <- eStep$S
  XB.hat <- VectorToMatrix(X %*% B.hat, n, p)
  XD.hat <- VectorToMatrix(X %*% D.hat, n, p)
  
  predicted <- exp(XB.hat + M.hat %*% t(C.hat) + 0.5 * (S.hat) %*% t(C.hat * C.hat))
  elbo1 <- out$elbo1 ; elbo2 <- out$elbo2 ; elbo3 <- out$elbo3
  elbo4 <- out$elbo4 ; elbo5 <- out$elbo5
  
  pred <- list(A = out$A, nu = nu, mu = mu, predicted = predicted)
  # elbo <- out$objective_values[length(out$objective_values)]
  elbo <- out$objective
  
  monitoring <- list(status = status, iterations = iter)
  
  
  res <- list(mStep = mStep,
              eStep = eStep,
              pred = pred,
              iter = iter,
              elboPath = elboPath,
              elbo = elbo,
              params.init = params.init,
              monitoring = monitoring,
              elbo1 = elbo1,
              elbo2 = elbo2,
              elbo3 = elbo3,
              elbo4 = elbo4,
              elbo5 = elbo5)
  
}





