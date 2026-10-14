# ELBO and parameter gradients (ZIP‑PLN variational objective)

Computes the Evidence Lower Bound (ELBO) and its gradients with respect
to the model and variational parameters for a zero‑inflated Poisson
log‑normal (ZIP‑PLN) latent factor model. This is a thin R wrapper
around a fast C++ implementation, with a small preprocessing step that
replaces `NA` values in `Y` by zeros before delegation.

## Usage

``` r
Elbo_grad(data, params, tolxi)
```

## Arguments

- data:

  A list with elements:

  `Y`

  :   Numeric `n x p` count matrix (may contain `NA`).

  `X`

  :   Numeric `n x d` design matrix for the abundance/log‑mean part.

  `R`

  :   Numeric/binary design matrix for the zero‑inflation (logit) part
      (same number of rows as `Y`); can be `NULL` if not used.

- params:

  A list with elements:

  `B`

  :   Matrix of regression coefficients for `X` (abundance).

  `D`

  :   Matrix of regression coefficients for `X` (zero‑inflation).

  `C`

  :   Loadings matrix for the latent factors (rank `q`).

  `M`

  :   Variational means of the latent factors.

  `S`

  :   Variational standard deviations (or log‑SDs) of the latent
      factors.

- tolxi:

  Numeric, convergence tolerance used for the variational bound on the
  logistic term.

## Value

A list containing at least:

- `elbo`:

  Scalar numeric, the ELBO value.

- `grad_B, grad_D, grad_C, grad_M, grad_S`:

  Gradients w.r.t. the corresponding parameters.

## Details

This function sets `NA` entries of `data$Y` to `0` internally (they are
not dropped) before calling the C++ backend `Elbo_grad_Rcpp`. Ensure the
dimensions of `X`, `R` and the parameter matrices are compatible with
`Y`.

## See also

`Elbo_grad_Rcpp` (backend C++).
