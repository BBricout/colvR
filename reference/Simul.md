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
set.seed(1)
n <- 30; p <- 5; d <- 2; q <- 2
X <- cbind(1, rnorm(n * p))                       # (n*p) x d design
theta <- list(B = matrix(c(1, 0.5)), D = matrix(c(1, -0.5)),
              C = matrix(rnorm(p * q, sd = 0.5), p, q))
sim <- Simul(X, theta, list(n = n, p = p, d = d, q = q))
str(sim)
#> List of 6
#>  $ Y     : int [1:30, 1:5] 2 3 4 0 0 0 0 0 0 0 ...
#>  $ W     : num [1:30, 1:2] -0.09818 0.56082 -1.18646 1.09678 -0.00534 ...
#>  $ U     : int [1:30, 1:5] 1 1 1 0 0 0 0 0 1 0 ...
#>  $ Z     : int [1:30, 1:5] 2 3 4 10 7 3 3 6 0 2 ...
#>  $ Lambda: num [1:30, 1:5] 1.68 4.24 2.6 9.23 5.31 ...
#>  $ mu    : num [1:30, 1:5] 0.687 1.092 0.582 1.798 1.165 ...
mean(sim$Y == 0)                                  # proportion of zeros
#> [1] 0.3733333
```
