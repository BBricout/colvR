# Variance-covariance matrix of theta

Computes \\V(\hat{\theta})\\ using a sandwich (robust) variance formula.

## Usage

``` r
V_theta(Y, X, fit, ginv_tol = 1e-12, verbose = TRUE)
```

## Arguments

- Y:

  An n x p matrix of counts.

- X:

  An (n\*p) x d design matrix, stacked by blocks of size n.

- fit:

  Output list returned by the estimator.

- ginv_tol:

  Numeric tolerance used by MASS::ginv() when a pseudo-inverse is
  needed.

- verbose:

  Logical; if TRUE, prints diagnostic messages when solve() fails or is
  unstable.

## Value

A symmetric variance-covariance matrix for theta.
