# Init_ZIP_q0

Parameters initialisation in the zero inflated case without the latent
factors

## Usage

``` r
Init_ZIP_q0(Y, X, q)
```

## Arguments

- Y:

  count matrix

- X:

  covariates

- q:

  size of the latent space

## Value

A list with elements:

- B:

  Matrix of Poisson regression coefficients (1 x d).

- D:

  Matrix of logistic regression coefficients (1 x d).
