## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup, include=FALSE-----------------------------------------------------
# Si torch était chargé par un profil/IDE, on le décharge proprement.
if ("torch" %in% loadedNamespaces()) try(unloadNamespace("torch"), silent = TRUE)

## -----------------------------------------------------------------------------
library(colvR)

## ----example------------------------------------------------------------------

data("fuligule_milouin")
Y <- fuligule_milouin$Y ; X <- fuligule_milouin$X
n <- nrow(Y) ; p <- ncol(Y) ; d <- ncol(X)

cat(mean(is.na(Y)), "is the proportion of missing entries in Y.\n")
cat(mean(Y == 0, na.rm = TRUE), "is the proportion of 0's in Y.\n")



## -----------------------------------------------------------------------------
q_grid <- 1:p

use_precomputed <- !identical(Sys.getenv("NOT_CRAN"), "true")

if (use_precomputed) {
  bics <- readRDS(system.file("extdata", "bics_fuligule.rds", package = "colvR"))
} else {
  bics <- select_model_bic(Y, X, Miss.ZIPLNPCA, q_grid)
  # (Optionnel) sauvegarder à nouveau si vous voulez mettre à jour le package :
  # saveRDS(bics, file = "inst/extdata/bics_fuligule.rds", compress = "xz")
}

# Selected q (robuste aux noms)
q <- if (!is.null(names(bics))) as.integer(names(bics)[which.max(bics)]) else q_grid[which.max(bics)]
q

## ----bics_plot, fig.cap="Model selection via BIC", fig.alt="Line plot of BIC versus the latent dimension q for the fuligule_milouin data; the selected q maximizes BIC."----
plot(q_grid, as.numeric(bics), type = "b",
     xlab = "Latent dimension q", ylab = "BIC")
# (optionnel) surligner q*
points(q, bics[which.max(bics)], pch = 19, cex = 1.2)
abline(v = q, lty = 2)

## -----------------------------------------------------------------------------
fit <- Miss.ZIPLNPCA(Y, X, q)

## -----------------------------------------------------------------------------
cov <- V_theta(Y, X, fit)
theta <- c(fit$mStep$gamma, fit$mStep$beta)
var <- diag(cov)[1:(2*d)]
Intervals <- IC(theta, var)

## ----ci_forest_plot, fig.cap="Regression coefficients with 95% confidence intervals", fig.alt="Horizontal dot-and-whisker plot showing gamma (presence) and beta (abundance) coefficients with their 95% confidence intervals; a vertical line marks zero."----
# Build a compact table of estimates and 95% CIs (no dependency on IC())
se <- sqrt(var)
par_names <- c(paste0("gamma_", seq_len(d)), paste0("beta_", seq_len(d)))
ci_tab <- data.frame(
  param = par_names,
  estimate = theta,
  lo = theta - 1.96 * se,
  hi = theta + 1.96 * se,
  group = rep(c("gamma (presence)", "beta (abundance)"), each = d),
  stringsAsFactors = FALSE
)

# Order for plotting: gamma first, then beta, top-to-bottom
ci_tab$param_pretty <- factor(ci_tab$param, levels = rev(ci_tab$param))

op <- par(mar = c(5, 10, 2, 2))
plot(
  x = ci_tab$estimate, y = as.numeric(ci_tab$param_pretty),
  xlab = "Coefficient (estimate with 95% CI)", ylab = "",
  yaxt = "n", pch = 19, xaxs = "i", col = ifelse(ci_tab$group == "gamma (presence)", "black", "gray30")
)
segments(ci_tab$lo, as.numeric(ci_tab$param_pretty),
         ci_tab$hi, as.numeric(ci_tab$param_pretty),
         lwd = 2, col = ifelse(ci_tab$group == "gamma (presence)", "black", "gray30"))
abline(v = 0, lty = 2)
axis(2, at = seq_along(levels(ci_tab$param_pretty)), labels = levels(ci_tab$param_pretty), las = 1)
legend("bottomright", inset = 0.01, bty = "n",
       legend = c("gamma (presence)", "beta (abundance)"),
       pch = 19, col = c("black", "gray30"))
par(op)


