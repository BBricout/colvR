# Small simulated dataset shared by the tests ----------------------------------

sim_data <- function(seed = 1, n = 30, p = 5, d = 2, q = 2, n_na = 20) {
  set.seed(seed)
  X <- cbind(1, rnorm(n * p))                       # (n*p) x d design
  theta <- list(B = matrix(c(1, 0.5)), D = matrix(c(1, -0.5)),
                C = matrix(rnorm(p * q, sd = 0.5), p, q))
  Y <- Simul(X, theta, list(n = n, p = p, d = d, q = q))$Y
  Y[sample(n * p, n_na)] <- NA                      # missing entries
  list(Y = Y, X = X, n = n, p = p, d = d, q = q)
}

# Fake fits (no optimisation), to test the criteria quickly --------------------

fake_zip_fit <- function(n, p, q, elbo = -500, S = 0.3, xi = 0.7) {
  list(elbo  = elbo,
       mStep = list(beta = matrix(c(1, 2)), gamma = matrix(c(1, 2))),
       eStep = list(S = matrix(S, n, q), xi = matrix(xi, n, p)),
       q     = q)
}

fake_pln_fit <- function(n, q, elbo = -500, S = 0.3) {
  list(elbo  = elbo,
       mStep = list(beta = matrix(c(1, 2))),
       eStep = list(S = matrix(S, n, q)),
       q     = q)
}
