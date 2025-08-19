#' Miss.ZIPLNPCA_3steps
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


Miss.ZIPLNPCA_3steps <- function(Y, # Matrice de comptage 
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
    config_vem <- list(maxiter = 1000, ftol = 1e-08, xtol = 1e-04, 
                       tolS = list(lower = 1e-04, upper = 1), tolxi = 1e-04)
  }
  
  if (is.null(config)){
    config <- list(algorithm = "MMA", backend = "nlopt", maxeval = 1000,
                   ftol_abs = 1e-8, xtol_abs = 1e-4, maxtime = -1, trace = 1, ftol_rel = 1e-15, xtol_rel = 1e-15)
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
  
  uBound <- c(rep(Inf, (2*d)+(p*q)+(n*q)), rep(tolS$upper, n*q))
  lBound <- c(rep(-Inf, (2*d)+(p*q)+(n*q)), rep(tolS$lower, n*q))
  config$lower_bounds <- lBound
  config$upper_bounds <- uBound
  
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
  
  ## VEM
  
  
  while((max(abs(params_new - params_old)) > xtol && abs(elbo_new - elbo_old) > ftol) && iter < maxiter){
    
    iter <- iter +1
    
    cat("Iteration:", iter, "ELBO:", elbo_new, "Diff params:", max(abs(params_new - params_old)), "Diff ELBO:", abs(elbo_new - elbo_old), "\n")
    
    params_old <- params_new
    elbo_old <- elbo_new
    
    OutM <- nlopt_optimize_ZIP_M(data, params, config, tolxi)
    params <- list(B = OutM$B, D = OutM$D, C = OutM$C,
                   M = OutM$M, S = OutM$S)
    
    grad <- Elbo_grad(data, params, tolxi = 1e-04)
    
    cat("gradB : ", mean((grad$gradB**2)), "gradD : ", mean((grad$gradD**2)), 
        "gradC : ", mean((grad$gradC**2)), "gradM : ", mean((grad$gradM**2)), 
        "gradS : ", mean((grad$gradS**2)), "\n")
    
    #print(OutM$monitoring)
    
    outVE <- nlopt_optimize_ZIP_VE_M(data, params, config, tolxi)
    params$M <- outVE$M 
    
    grad <- Elbo_grad(data, params, tolxi = 1e-04)
    
    cat("gradB : ", mean((grad$gradB**2)), "gradD : ", mean((grad$gradD**2)), 
        "gradC : ", mean((grad$gradC**2)), "gradM : ", mean((grad$gradM**2)), 
        "gradS : ", mean((grad$gradS**2)), "\n")
    
    #print(outVE$monitoring)
    
    # outM <- nlopt_optimize_ZIP_M(data, params, config, tolxi)
    # params <- list(B = OutM$B, D = OutM$D, C = OutM$C,
    #                M = OutM$M, S = OutM$S)
    # 
    # grad <- Elbo_grad(data, params, tolxi = 1e-04)
    # 
    # cat("gradB : ", mean((grad$gradB**2)), "gradD : ", mean((grad$gradD**2)), 
    #     "gradC : ", mean((grad$gradC**2)), "gradM : ", mean((grad$gradM**2)), 
    #     "gradS : ", mean((grad$gradS**2)), "\n")
    
    #print(outGamma$monitoring)
    
    outVE <- nlopt_optimize_ZIP_VE_S(data, params, config, tolxi)
    params$S <- outVE$S
    
    grad <- Elbo_grad(data, params, tolxi = 1e-04)
    
    cat("gradB : ", mean((grad$gradB**2)), "gradD : ", mean((grad$gradD**2)), 
        "gradC : ", mean((grad$gradC**2)), "gradM : ", mean((grad$gradM**2)), 
        "gradS : ", mean((grad$gradS**2)), "\n")
    
    #print(outVE$monitoring)
    
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





