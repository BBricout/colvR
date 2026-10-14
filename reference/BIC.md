# BIC from ELBO for a ZIP-PLN latent factor model

Computes a Bayesian Information Criterion (BIC)-like score from the
Evidence Lower Bound (ELBO) of a fitted model. The penalty accounts for
the number of free parameters in a latent factor model with latent
dimension `q`: \$\$ \mathrm{BIC} = \mathrm{ELBO} -
\frac{\log(n)}{2}\Big\[q\\p - (q-1)/2\\ + p\\d\Big\], \$\$ where \\n\\
is the number of rows (sites/samples), \\p\\ the number of columns
(species/variables), and \\d\\ the number of covariates.

## Usage

``` r
BIC(Y, X, fit, q)
```

## Arguments

- Y:

  Numeric `n x p` count matrix.

- X:

  Numeric `n x d` design matrix of covariates (first column typically an
  intercept).

- fit:

  A fitted object (list) containing at least `$elbo`, the maximized ELBO
  value for the model.

- q:

  Integer, size of the latent space (rank).

## Value

A single numeric value: the BIC score (higher is better in this
ELBO-based convention).

## Details

This criterion mirrors the usual BIC structure but replaces the
log-likelihood with the ELBO and uses the parameter count appropriate
for a rank-`q` latent factor structure.

## See also

[`BIC`](https://rdrr.io/r/stats/AIC.html)
