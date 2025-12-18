
<!-- README.md is generated from README.Rmd. Please edit that file -->

Documentation: <https://BBricout.github.io/colvR/>

# colvR

<!-- badges: start -->

[![R-CMD-check](https://github.com/BBricout/colvR/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/BBricout/colvR/actions/workflows/R-CMD-check.yaml)

[![R-CMD-check](https://github.com/BBricout/colvR/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/BBricout/colvR/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

colvR provides modeling tools for count data — especially overdispersed
data with many zeros — and includes missing-data imputation. It also
exposes fast Rcpp/Armadillo utilities for matrix–vector conversions and
low-rank ZIP/PLN optimization.

## Installation

From GitHub:

``` r
# install.packages("devtools")
devtools::install_github("BBricout/colvR")

library(colvR)
```

When on CRAN, you’ll be able to do: install.packages(“colvR”)

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(colvR)
#> 
#> Attachement du package : 'colvR'
#> L'objet suivant est masqué depuis 'package:stats':
#> 
#>     BIC

n <- 1500 ; p <- 30 ; q <- 4 ; d <- 4
dim <- list(n = n, p = p, d = d, q = q)
X <- cbind(rep(1, n*p), rnorm(n*p), rep(rnorm(n), p))
trend <- rep(1:p, each = n)
X <- cbind(X, scale(trend))
B <- c(2, 0.7, 0.5, -0.6) ; D <- c(-0.5, 0.5, 0.4, -0.4) ; C <- matrix(rnorm(p*q)/2, nrow = p, ncol = q)
theta <- list(B = B, D = D, C = C)

sim <- Simul(X, theta, dim) ; Y <- sim$Y

fit <- Miss.ZIPLNPCA(Y, X, q)
```
