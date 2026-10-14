# Model selection by BIC over candidate latent dimensions

For each value in `latents`, fits a model via `fun(Y, X, q)` and
computes the corresponding BIC. Returns the vector of BIC values in the
same order as `latents`.

## Usage

``` r
select_model_bic(Y, X, fun, latents)
```

## Arguments

- Y:

  Response matrix (n x p).

- X:

  Covariate matrix (n x d).

- fun:

  Fitting function called as `fun(Y, X, q)` that returns a fit object.

- latents:

  Integer/numeric vector of candidate latent dimensions.

## Value

A numeric vector of BIC values, one per element of `latents`, in the
same order as `latents`.

## Details

A progress bar is displayed via pbapply. The `BIC` function used must be
available in your namespace (e.g.,
[`colvR::BIC`](https://bbricout.github.io/colvR/reference/BIC.md) if you
define one, or [`stats::BIC`](https://rdrr.io/r/stats/AIC.html) if you
call it explicitly).
