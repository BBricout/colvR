pkgname <- "colvR"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
base::assign(".ExTimings", "colvR-Ex.timings", pos = 'CheckExEnv')
base::cat("name\tuser\tsystem\telapsed\n", file=base::get(".ExTimings", pos = 'CheckExEnv'))
base::assign(".format_ptime",
function(x) {
  if(!is.na(x[4L])) x[1L] <- x[1L] + x[4L]
  if(!is.na(x[5L])) x[2L] <- x[2L] + x[5L]
  options(OutDec = '.')
  format(x[1L:3L], digits = 7L)
},
pos = 'CheckExEnv')

### * </HEADER>
library('colvR')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("IC")
### * IC

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: IC
### Title: 95% interval for a theta 95% confidence interval for theta
### Aliases: IC

### ** Examples

IC(1, 0.04)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("IC", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("Init")
### * Init

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: Init
### Title: Initialize parameters for a PLN-PCA with missing data
### Aliases: Init

### ** Examples

set.seed(1)
n <- 30; p <- 8; d <- 2; q <- 2
Y <- matrix(rpois(n * p, 2), n, p)
X <- cbind(1, rnorm(n))          # rowwise design
init <- Init(Y, X, q)
str(init)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("Init", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("Init_ZIP")
### * Init_ZIP

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: Init_ZIP
### Title: Initialize parameters for the ZIP case (zero-inflated)
### Aliases: Init_ZIP

### ** Examples

set.seed(1)
n <- 30; p <- 10; d <- 3; q <- 2
Y <- matrix(rpois(n*p, 2), n, p)
X <- cbind(1, rnorm(n*p), rnorm(n*p))  # (n*p) x d, vectorized design
init <- Init_ZIP(Y, X, q)
str(init)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("Init_ZIP", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("Miss.PLNPCA")
### * Miss.PLNPCA

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: Miss.PLNPCA
### Title: PLN-PCA with missing data (variational fit + imputation)
### Aliases: Miss.PLNPCA

### ** Examples

## Not run: 
##D set.seed(1)
##D n <- 40; p <- 12; d <- 2; q <- 2
##D Y <- matrix(rpois(n*p, 2), n, p)
##D Y[sample(length(Y), 20)] <- NA
##D X <- cbind(1, rnorm(n))      # rowwise design
##D 
##D fit <- Miss.PLNPCA(
##D   Y = Y, X = X, q = q,
##D   O = matrix(0, n, p),
##D   w = rep(1, n),
##D   params = NULL,
##D   config = NULL
##D )
##D str(fit$pred$A)     # expected counts (n x p)
##D fit$elbo            # final ELBO
## End(Not run)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("Miss.PLNPCA", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("Miss.ZIPLNPCA")
### * Miss.ZIPLNPCA

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: Miss.ZIPLNPCA
### Title: ZI‑PLN-PCA with missing data (variational fit + imputation)
### Aliases: Miss.ZIPLNPCA

### ** Examples

## Not run: 
##D set.seed(1)
##D n <- 40; p <- 12; d <- 3
##D q <- 2
##D Y <- matrix(rpois(n*p, 2), n, p)
##D Y[sample(length(Y), 25)] <- NA
##D # Vectorized design (n*p) x d:
##D X <- cbind(1, rnorm(n*p), rnorm(n*p))
##D 
##D fit <- Miss.ZIPLNPCA(Y = Y, X = X, q = q)
##D str(fit$mStep)
##D str(fit$eStep)
##D image(log1p(fit$imputed))  # quick look at imputed counts
##D 
##D # Rank-0 (ZIP regression without latent factors):
##D fit0 <- Miss.ZIPLNPCA(Y = Y, X = X, q = 0)
##D fit0$eStep$xi[1:3, 1:3]
## End(Not run)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("Miss.ZIPLNPCA", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("Miss.ZIPLNPCA.logS")
### * Miss.ZIPLNPCA.logS

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: Miss.ZIPLNPCA.logS
### Title: ZI‑PLN-PCA with missing data using log(S) parametrization
### Aliases: Miss.ZIPLNPCA.logS

### ** Examples

## Not run: 
##D set.seed(1)
##D n <- 40; p <- 12; d <- 3; q <- 2
##D Y <- matrix(rpois(n*p, 2), n, p); Y[sample(length(Y), 20)] <- NA
##D X <- cbind(1, rnorm(n*p), rnorm(n*p))  # (n*p) x d
##D 
##D fit <- Miss.ZIPLNPCA.logS(Y = Y, X = X, q = q, tolLogS = 2)  # cap logS
##D fit$elbo
##D str(fit$eStep$S)
## End(Not run)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("Miss.ZIPLNPCA.logS", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("Miss.ZIPLNPCA_Steps")
### * Miss.ZIPLNPCA_Steps

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: Miss.ZIPLNPCA_Steps
### Title: ZI‑PLN-PCA (missing data) with parameterwise optimization steps
### Aliases: Miss.ZIPLNPCA_Steps

### ** Examples

## Not run: 
##D set.seed(1)
##D n <- 50; p <- 15; d <- 3; q <- 2
##D Y <- matrix(rpois(n*p, 2), n, p); Y[sample(length(Y), 30)] <- NA
##D X <- cbind(1, rnorm(n*p), rnorm(n*p))  # (n*p) x d, vectorized design
##D 
##D fit <- Miss.ZIPLNPCA_Steps(Y, X, q)
##D fit$elbo
##D plot(fit$elboPath, type = "l", xlab = "outer iter", ylab = "ELBO")
##D str(fit$mStep); str(fit$eStep)
## End(Not run)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("Miss.ZIPLNPCA_Steps", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("Miss.ZIPLNPCA_VE")
### * Miss.ZIPLNPCA_VE

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: Miss.ZIPLNPCA_VE
### Title: ZI‑PLN-PCA (missing data) — variational E‑step solver
### Aliases: Miss.ZIPLNPCA_VE

### ** Examples

## Not run: 
##D set.seed(1)
##D n <- 40; p <- 12; d <- 3; q <- 2
##D Y <- matrix(rpois(n*p, 2), n, p); Y[sample(length(Y), 20)] <- NA
##D X <- cbind(1, rnorm(n*p), rnorm(n*p))  # (n*p) x d
##D 
##D fit <- Miss.ZIPLNPCA_VE(Y = Y, X = X, q = q)
##D fit$elbo
##D str(fit$mStep); str(fit$eStep)
## End(Not run)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("Miss.ZIPLNPCA_VE", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("Predictions.marginales")
### * Predictions.marginales

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: Predictions.marginales
### Title: Prediction intervals (marginal) for ZIPLNPCA-like model
### Aliases: Predictions.marginales

### ** Examples

## Not run: 
##D n <- 5; p <- 3; d <- 2
##D set.seed(1)
##D Y <- matrix(rpois(n*p, 2), n, p)
##D X <- matrix(rnorm(n*p*d), n*p, d)
##D fit <- list(
##D   mStep = list(beta = rnorm(d), gamma = rnorm(d), C = matrix(rnorm(p*2), p, 2)),
##D   eStep = list(M = matrix(0, n, 2), S = matrix(1, n, 2), xi = matrix(0.5, n, p))
##D )
##D out <- Predictions.marginales(Y, X, fit, MC = 100)
##D str(out$lower)
## End(Not run)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("Predictions.marginales", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("Simul")
### * Simul

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: Simul
### Title: Simulate zero-inflated Poisson log-normal PCA data
### Aliases: Simul

### ** Examples

## Not run: 
##D n <- 5; p <- 4; d <- 3; q <- 2
##D X <- matrix(rnorm(n*p*d), n*p, d)
##D theta <- list(B = matrix(rnorm(d)), D = matrix(rnorm(d)), C = matrix(rnorm(p*q), p, q))
##D dim  <- list(n = n, p = p, d = d, q = q)
##D out <- Simul(X, theta, dim)
##D str(out)
## End(Not run)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("Simul", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("covmat")
### * covmat

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: covmat
### Title: Wrapper for 'lori::covmat'
### Aliases: covmat

### ** Examples

if (requireNamespace("lori", quietly = TRUE)) {
  set.seed(1)
  n <- 3; p <- 2
  R <- matrix(rnorm(n * 2), nrow = n, ncol = 2)  # site-level (d_R = 2)
  C <- matrix(rnorm(p * 2), nrow = p, ncol = 2)  # year-level (d_C = 2)
  X <- covmat(n, p, R = R, C = C)
  dim(X)  # 6 x 4
}



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("covmat", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("fuligule_milouin")
### * fuligule_milouin

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: fuligule_milouin
### Title: Fuligule milouin example dataset
### Aliases: fuligule_milouin
### Keywords: datasets

### ** Examples

data(fuligule_milouin, package = "colvR")
str(fuligule_milouin, max.level = 1)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("fuligule_milouin", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
