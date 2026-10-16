# ZI‑PLN-PCA with missing data (variational fit + imputation)

Fits a zero‑inflated Poisson log‑normal (ZI‑PLN-PCA) latent factor model
to a count matrix with missing values. The routine optimizes a
variational objective with NLOpt backends and returns parameter
estimates, variational parameters and an imputed matrix.

## Usage

``` r
Miss.ZIPLNPCA(Y, X, q, params = NULL, config = NULL, tolS = NULL, tolxi = NULL)
```

## Arguments

- Y:

  Numeric `n x p` count matrix. May contain `NA`.

- X:

  Numeric design matrix with `n*p` rows and `d` columns, aligned with
  `vec(Y)` (column‑wise vectorization).

- q:

  Integer, latent rank (dimension of the latent space).

- params:

  Optional list of initial parameters. If `NULL`, they are initialized
  internally via
  [`Init_ZIP_q0`](https://bbricout.github.io/colvR/reference/Init_ZIP_q0.md)
  (when `q = 0`) or
  [`Init_ZIP`](https://bbricout.github.io/colvR/reference/Init_ZIP.md)
  (when `q > 0`).

- config:

  Optional list of optimizer controls. If `NULL`, a default
  configuration is used (NLOpt backend, MMA algorithm, sensible
  tolerances).

- tolS:

  List with numeric bounds for the variational scales `S`: elements
  `lower` and `upper`. Defaults to `list(lower = 0, upper = 1)`.

- tolxi:

  Numeric tolerance for the Jaakkola‑type \\\xi\\ updates in the
  logistic bound (default `1e-4`).

## Value

A list with components:

- `mStep`:

  List of model parameters. For `q = 0`: `gamma` (`d x 1`) and `beta`
  (`d x 1`). For `q > 0`: same plus the loading matrix `C` (`p x q`).

- `eStep`:

  List of variational parameters. For `q = 0`: only `xi` (`n x p`). For
  `q > 0`: `M` (`n x q`), `S` (`n x q`) and `xi` (`n x p`).

- `pred`:

  List with predictors and expected counts: `mu` (`n x p`, abundance
  mean), `nu` (`n x p`, zero‑inflation mean), `A` (`n x p`, PLN
  expectation), `predicted` (`n x p`, ZIP expectation for all entries:
  \\\xi \odot A\\, with \\A = \exp( XB + M C^\top + 0.5\\ S\\ (C\odot
  C)^\top )\\, `S` being the variational variances).

- `imputed`:

  `n x p` matrix equal to `predicted` at missing entries of `Y`, and `Y`
  elsewhere (ZIP expectation).

- `iter`:

  Integer, number of iterations.

- `elboPath`:

  Numeric vector of objective (ELBO) values over iterations.

- `elbo`:

  Final ELBO value.

- `params.init`:

  Parameters as recorded from the backend call.

- `monitoring`:

  Optimizer diagnostics/logs from the backend.

- `gradB, gradD, gradC, gradM, gradS`:

  Gradients of the ELBO with respect to the corresponding parameters,
  obtained via
  [`Elbo_grad`](https://bbricout.github.io/colvR/reference/Elbo_grad.md).

## Details

This function supports latent rank `q >= 0`. When `q = 0`, it reduces to
a ZIP regression (no latent factors). When `q > 0`, it fits a ZIP‑PLN
factor model.

Internally, a binary mask `R = 1_{observed}(Y)` is created; a copied
matrix `Y.na` sets missing entries to zero for objective evaluation. The
optimizer bounds for `S` are taken from `tolS`. Ensure that the
vectorization and parameter stacking used in the optimizer are
consistent with the shapes listed above.

## See also

[`Init_ZIP`](https://bbricout.github.io/colvR/reference/Init_ZIP.md),
[`Init_ZIP_q0`](https://bbricout.github.io/colvR/reference/Init_ZIP_q0.md),
[`Elbo_grad`](https://bbricout.github.io/colvR/reference/Elbo_grad.md),
[`VectorToMatrix`](https://bbricout.github.io/colvR/reference/VectorToMatrix.md)

## Examples

``` r
set.seed(1)
n <- 30; p <- 5; d <- 2; q <- 2
X <- cbind(1, rnorm(n * p))                       # (n*p) x d design
theta <- list(B = matrix(c(1, 0.5)), D = matrix(c(1, -0.5)),
              C = matrix(rnorm(p * q, sd = 0.5), p, q))
Y <- Simul(X, theta, list(n = n, p = p, d = d, q = q))$Y
Y[sample(n * p, 20)] <- NA                        # missing entries

fit <- Miss.ZIPLNPCA(Y, X, q)
str(fit$mStep)
#> List of 3
#>  $ gamma: num [1:2, 1] 0.846 -0.62
#>  $ beta : num [1:2, 1] 0.866 0.558
#>  $ C    : num [1:5, 1:2] 0.4901 -0.2339 0.0273 0.1045 -1.0393 ...
fit$imputed[is.na(Y)][1:5]                        # imputed values
#> [1] 3.056266 2.878426 2.420737 1.299346 2.493428
```
