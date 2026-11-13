# Model selection over candidate latent dimensions (BIC or ICL)

For each value in `latents`, fits a model with `fun` and computes the
chosen criterion
([`BIC`](https://bbricout.github.io/colvR/reference/BIC.md) or
[`ICL`](https://bbricout.github.io/colvR/reference/ICL.md)). Returns the
criterion values, the latent dimension that maximizes the criterion and
the corresponding fit.

## Usage

``` r
select_model_bic(
  Y,
  X,
  fun,
  latents,
  model = c("ZIPLNPCA", "PLNPCA"),
  criterion = c("BIC", "ICL")
)
```

## Arguments

- Y:

  Numeric `n x p` count matrix (NAs allowed).

- X:

  Design matrix, `(n*p) x d`.

- fun:

  Fitting function, called as `fun(Y = Y, X = X, q = q)` (e.g.
  `Miss.ZIPLNPCA` or `Miss.PLNPCA`).

- latents:

  Integer vector of candidate latent dimensions.

- model:

  `"ZIPLNPCA"` (default) or `"PLNPCA"`; must match `fun`. Passed to
  [`BIC`](https://bbricout.github.io/colvR/reference/BIC.md) and
  [`ICL`](https://bbricout.github.io/colvR/reference/ICL.md).

- criterion:

  `"BIC"` (default) or `"ICL"`.

## Value

A list with elements:

- `criterion`: the criterion used.

- `values`: numeric vector of criterion values, named by `latents`.

- `q_star`: the latent dimension maximizing the criterion.

- `fit_star`: the fit obtained with `q_star`.

## Details

A progress bar is displayed via pbapply. The function checks that
`model` matches the fits returned by `fun`: a ZIP-PLN-PCA fit contains
`mStep$gamma`, a PLN-PCA fit does not.

## See also

[`BIC`](https://bbricout.github.io/colvR/reference/BIC.md),
[`ICL`](https://bbricout.github.io/colvR/reference/ICL.md)

## Examples

``` r
set.seed(1)
n <- 30; p <- 5; d <- 2; q <- 2
X <- cbind(1, rnorm(n * p))                       # (n*p) x d design
theta <- list(B = matrix(c(1, 0.5)), D = matrix(c(1, -0.5)),
              C = matrix(rnorm(p * q, sd = 0.5), p, q))
Y <- Simul(X, theta, list(n = n, p = p, d = d, q = q))$Y
Y[sample(n * p, 20)] <- NA                        # missing entries

# \donttest{
sel <- select_model_bic(Y, X, Miss.ZIPLNPCA, latents = 1:3)
sel$values
#>         1         2         3 
#> -265.5807 -267.4258 -272.9657 
sel$q_star
#> [1] 1
fit <- sel$fit_star

sel_icl <- select_model_bic(Y, X, Miss.ZIPLNPCA, latents = 1:3,
                            criterion = "ICL")
sel_icl$q_star
#> [1] 1
# }
```
