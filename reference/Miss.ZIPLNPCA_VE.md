# ZI‑PLN-PCA (missing data) — variational E‑step solver

Fits a zero‑inflated Poisson log‑normal (ZI‑PLN-PCA) latent factor model
on a count matrix with missing values using a single call to a
variational optimization backend (VE). Returns model/variational
parameters and expected Poisson means suitable for imputation and
prediction.

## Usage

``` r
Miss.ZIPLNPCA_VE(Y, X, q, params = NULL, config_vem = NULL, config = NULL)
```

## Arguments

- Y:

  Numeric `n x p` count matrix. May contain `NA`.

- X:

  Numeric design matrix with `n*p` rows and `d` columns, aligned with
  `vec(Y)` (vectorization by columns).

- q:

  Integer, latent rank (dimension of the latent space).

- params:

  Optional list of initial parameters. If `NULL`,
  [`Init_ZIP`](https://bbricout.github.io/colvR/reference/Init_ZIP.md)
  is called.

- config_vem:

  List of outer VE controls (stopping and bounds), with fields:

  `maxiter`

  :   Maximum iterations (not always used by this backend, default
      `10000`).

  `ftol`

  :   ELBO tolerance (default `1e-08`).

  `xtol`

  :   Parameter tolerance (default `1e-04`).

  `tolS`

  :   List with `lower`, `upper` bounds for `S` (default
      `list(lower = 1e-4, upper = 1)`).

  `tolxi`

  :   Tolerance for the Jaakkola‑type \\\xi\\ updates (default `1e-04`).

- config:

  List of NLOpt controls for the backend (algorithm, tolerances,
  `maxeval`, etc.). If `NULL`, sensible defaults are provided.

## Value

A list with components:

- `mStep`:

  Model parameters: `gamma` (`d x 1`), `beta` (`d x 1`), loadings `C`
  (`p x q`).

- `eStep`:

  Variational parameters: `M` (`n x q`), `S` (`n x q`), `xi` (`n x p`).

- `pred`:

  List with predictors and expected counts: `mu` (`n x p`), `nu`
  (`n x p`), backend mean `A` (`n x p`), and `predicted` (`n x p`, ZIP
  expectation for all entries: \\\xi \odot A\\, with \\A = \exp( XB + M
  C^\top + 0.5\\ S\\ (C\odot C)^\top )\\, `S` being the variational
  variances).

- `elbo`:

  Final ELBO value.

- `params.init`:

  Initial parameters used.

- `monitoring`:

  Backend diagnostics (if provided).

- `elbo1, elbo2, elbo3, elbo4, elbo5`:

  Decomposition terms of the ELBO (as returned by the backend).

## Details

Missing entries are handled via a binary mask `R = 1_{observed}(Y)`, and
a working copy `Y.na` with zeros at missing locations for the objective
evaluation. Box constraints on `S` are injected into
`config$lower_bounds` and `config$upper_bounds` using `config_vem$tolS`.

**Predicted mean:** ici, la recomposition côté R utilise \$\$
\exp\\\big( X B + M C^\top + 0.5\\ S \\(C\odot C)^\top \big). \$\$
D'autres fonctions du package emploient \\0.5\\(S\odot S)\\(C\odot
C)^\top\\. Vérifie et harmonise selon la paramétrisation attendue par
ton backend.

## Examples

``` r
set.seed(1)
n <- 30; p <- 5; d <- 2; q <- 2
X <- cbind(1, rnorm(n * p))                       # (n*p) x d design
theta <- list(B = matrix(c(1, 0.5)), D = matrix(c(1, -0.5)),
              C = matrix(rnorm(p * q, sd = 0.5), p, q))
Y <- Simul(X, theta, list(n = n, p = p, d = d, q = q))$Y
Y[sample(n * p, 20)] <- NA                        # missing entries

fit <- Miss.ZIPLNPCA_VE(Y, X, q)
fit$elbo
#> [1] -282.2486
str(fit$eStep)
#> List of 3
#>  $ M : num [1:30, 1:2] 0.135 0.504 -0.746 0.702 0.114 ...
#>  $ S : num [1:30, 1:2] 0.843 0.367 0.22 0.502 0.888 ...
#>  $ xi: num [1:30, 1:5] 1 0.5824 1 0.0546 0.1088 ...
```
