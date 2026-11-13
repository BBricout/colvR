# ZI‑PLN-PCA with missing data using log(S) parametrization

Fits a zero‑inflated Poisson log‑normal (ZI‑PLN-PCA) latent factor model
to a count matrix with missing values, using a variational objective and
an **alternative parametrization** on the variational scales: \$\$
\code{log(S)} \text{ instead of } S. \$\$ This guarantees positivity of
\\S\\ and allows simple box constraints on \\`log(S)` \\ via `tolLogS`.

## Usage

``` r
Miss.ZIPLNPCA.logS(
  Y,
  X,
  q,
  params = NULL,
  config = NULL,
  tolxi = NULL,
  tolLogS = NULL
)
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

  Optional list of initial parameters. If `NULL`, they are initialized
  by
  [`Init_ZIP`](https://bbricout.github.io/colvR/reference/Init_ZIP.md).
  If provided, it must include a positive `S`; the function internally
  adds `logS <- log(S)`.

- config:

  Optional list of optimizer controls. If `NULL`, defaults to
  `PLNPCA_param()$config_optim`.

- tolxi:

  Numeric tolerance for the Jaakkola‑type \\\xi\\ updates in the
  logistic bound (default `1e-4`).

- tolLogS:

  Numeric upper bound applied to `logS` (box constraint). Defaults to
  `Inf`. Use this to avoid excessively large variational scales when
  needed.

## Value

A list with components:

- `mStep`:

  Model parameters: `gamma` (`d x 1`), `beta` (`d x 1`), and loadings
  `C` (`p x q`).

- `eStep`:

  Variational parameters: means `M` (`n x q`), scales `S` (`n x q`), and
  logistic bound parameters `xi` (`n x p`).

- `pred`:

  List with predictors and expected counts: `mu` (`n x p`), `nu`
  (`n x p`), backend Poisson mean `A` (`n x p`), and `predicted`
  (`n x p`, ZIP expectation for all entries: \\\xi \odot A\\, with \\A =
  \exp( XB + M C^\top + 0.5\\ S\\ (C\odot C)^\top )\\, `S` being the
  variational variances).

- `imputed`:

  `n x p` matrix equal to `predicted` at missing entries of `Y`, and `Y`
  elsewhere.

- `iter`:

  Number of iterations; `elboPath` (trajectory of ELBO); `elbo` (final
  ELBO).

- `params.init`:

  Initial parameters passed to the backend (with `logS` added).

- `monitoring`:

  Optimizer diagnostics/logs.

- `gradB, gradD, gradC, gradM, gradS`:

  ELBO gradients w.r.t. corresponding parameters (via
  [`Elbo_grad`](https://bbricout.github.io/colvR/reference/Elbo_grad.md)).

## Details

Missing entries are handled by a mask `R = 1_{observed}(Y)`; a working
matrix `Y.na` sets missings to 0 for the objective evaluation. Box
constraints are set on `logS` via `config$upper_bounds`, using
`tolLogS`. Make sure `X` matches `vec(Y)`.

## See also

[`Miss.ZIPLNPCA`](https://bbricout.github.io/colvR/reference/Miss.ZIPLNPCA.md)
for the `S` parametrization,
[`Init_ZIP`](https://bbricout.github.io/colvR/reference/Init_ZIP.md),
[`Elbo_grad`](https://bbricout.github.io/colvR/reference/Elbo_grad.md)

## Examples

``` r
set.seed(1)
n <- 30; p <- 5; d <- 2; q <- 2
X <- cbind(1, rnorm(n * p))                       # (n*p) x d design
theta <- list(B = matrix(c(1, 0.5)), D = matrix(c(1, -0.5)),
              C = matrix(rnorm(p * q, sd = 0.5), p, q))
Y <- Simul(X, theta, list(n = n, p = p, d = d, q = q))$Y
Y[sample(n * p, 20)] <- NA                        # missing entries

fit <- Miss.ZIPLNPCA.logS(Y, X, q, tolLogS = 2)   # cap logS
fit$elbo
#> [1] -244.7575
str(fit$eStep$S)
#>  num [1:30, 1:2] 0.591 0.314 0.29 0.385 0.606 ...
```
