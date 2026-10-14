# Simulate zero-inflated Poisson log-normal PCA data

Generates simulated observations `Y` under a ZIP-LN-PCA structure from
parameters in `theta` and dimensions in `dim`.

## Usage

``` r
Simul(X, theta, dim)
```

## Arguments

- X:

  An \\(n \times p) \cdot d\\ design matrix stacked so that
  `VectorToMatrix(X %*% B, n, p)` produces the mean matrix `mu`.

- theta:

  List of model parameters with components:

  - `B`: coefficient matrix (d x 1 or compatible)

  - `D`: zero-inflation coefficient matrix (d x 1 or compatible)

  - `C`: loading matrix (p x q)

- dim:

  List of dimensions with entries `n`, `p`, `d`, `q`.

## Value

A list with elements:

- `Y`: simulated counts (n x p)

- `W`: latent scores (n x q)

- `U`: zero-inflation indicators (n x p)

- `Z`: Poisson draws before zero-inflation (n x p)

- `Lambda`: Poisson intensities (n x p)

- `mu`: mean matrix (n x p)

## Details

Relies on
[`VectorToMatrix()`](https://bbricout.github.io/colvR/reference/VectorToMatrix.md)
being available in the package.

## Examples

``` r
if (FALSE) { # \dontrun{
n <- 5; p <- 4; d <- 3; q <- 2
X <- matrix(rnorm(n*p*d), n*p, d)
theta <- list(B = matrix(rnorm(d)), D = matrix(rnorm(d)), C = matrix(rnorm(p*q), p, q))
dim  <- list(n = n, p = p, d = d, q = q)
out <- Simul(X, theta, dim)
str(out)
} # }
```
