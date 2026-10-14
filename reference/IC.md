# 95% interval for a theta 95% confidence interval for theta

Computes the 95% confidence interval for `theta`.

## Usage

``` r
IC(theta, var)
```

## Arguments

- theta:

  Numeric vector of estimates.

- var:

  Numeric vector of associated variances (same length as `theta`).

## Value

A numeric vector of length 2: lower and upper bounds.

## Examples

``` r
IC(1, 0.04)
#> [1] 0.608 1.392
```
