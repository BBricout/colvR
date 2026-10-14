# ZI‑PLN-PCA (missing data) with parameterwise optimization steps

Fits a zero‑inflated Poisson log‑normal (ZI‑PLN-PCA) latent factor model
to a count matrix with missing values using a **blockwise** optimization
strategy. At each outer iteration, parameters are updated by activating
one block among {B, D, C, M, S} while keeping the others fixed, via an
NLOpt backend. The loop stops when both parameter and ELBO changes are
below thresholds or when `maxiter` is reached.

## Usage

``` r
Miss.ZIPLNPCA_Steps(Y, X, q, params = NULL, config_vem = NULL, config = NULL)
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

  Optional list of initial parameters. If `NULL`,
  [`Init_ZIP`](https://bbricout.github.io/colvR/reference/Init_ZIP.md)
  is used.

- config_vem:

  List of controls for the outer blockwise loop (variational EM‑like),
  with fields:

  `maxiter`

  :   Maximum number of outer iterations (default `1e6`).

  `ftol`

  :   ELBO tolerance; stop if `|ELBO_{t}-ELBO_{t-1}| <= ftol`.

  `xtol`

  :   Parameter tolerance; stop if `max|theta_{t}-theta_{t-1}| <= xtol`.

  `tolS`

  :   List with `lower`, `upper` bounds for `S`.

  `tolxi`

  :   Tolerance used in the Jaakkola‑type \\\xi\\ updates for the
      logistic bound.

- config:

  List of controls for the NLOpt backend (per‑block inner solve), e.g.
  `algorithm`, `maxeval`, `ftol_abs`, `xtol_abs`, `ftol_rel`,
  `xtol_rel`, `trace`, `maxtime`, as well as `upper_bounds`,
  `lower_bounds` (filled here from `tolS` and dimensions).

## Value

A list with components:

- `mStep`:

  Model parameters: `gamma` (`d x 1`), `beta` (`d x 1`), loadings `C`
  (`p x q`).

- `eStep`:

  Variational parameters: `M` (`n x q`), `S` (`n x q`), logistic bound
  parameters `xi` (`n x p`).

- `pred`:

  List with predictors and expected counts: `mu` (`n x p`), `nu`
  (`n x p`), backend mean `A` (`n x p`), and `predicted` recomputed in
  R.

- `imputed`:

  `n x p` matrix equal to `xi * A` at missing entries of `Y`, and `Y`
  elsewhere.

- `iter`:

  Number of outer iterations performed.

- `elboPath`:

  Numeric vector of ELBO values across outer iterations.

- `elbo`:

  Final ELBO value.

- `params.init`:

  Parameters used for initialization.

- `monitoring`:

  List with `status` (stopping reason) and `iterations`.

- `gradB, gradD, gradC, gradM, gradS`:

  ELBO gradients w.r.t. each parameter block (from
  [`Elbo_grad`](https://bbricout.github.io/colvR/reference/Elbo_grad.md))
  at the solution.

## Details

Missing entries are handled through a binary mask and a working copy of
`Y` with zeros at missing locations for the objective evaluation.

A binary mask `R = 1_{observed}(Y)` is built; a working matrix `Y.na`
replaces missings by 0 for the objective. Box constraints for `S` are
set using `config_vem$tolS` and injected into `config$upper_bounds` and
`config$lower_bounds`. Each block update calls the backend
`nlopt_optimize_ZIP_Steps(data, params, config, tolxi, active_blocks)`
with a single active block at a time.

**Predicted mean:** in this implementation, the Poisson mean used for
the R‑side recomputation is \$\$ \exp\\\big( \mu + M C^\top + 0.5\\ S
\\(C\odot C)^\top \big), \$\$ where \\\mu = X B\\ reshaped into `n x p`.
Note that other functions in le package emploient \\0.5\\(S\odot
S)\\(C\odot C)^\top\\. Vérifie que la paramétrisation voulue est
cohérente avec le backend.

## See also

[`Miss.ZIPLNPCA`](https://bbricout.github.io/colvR/reference/Miss.ZIPLNPCA.md),
[`Miss.ZIPLNPCA.logS`](https://bbricout.github.io/colvR/reference/Miss.ZIPLNPCA.logS.md),
[`Init_ZIP`](https://bbricout.github.io/colvR/reference/Init_ZIP.md),
[`Elbo_grad`](https://bbricout.github.io/colvR/reference/Elbo_grad.md)

## Examples

``` r
if (FALSE) { # \dontrun{
set.seed(1)
n <- 50; p <- 15; d <- 3; q <- 2
Y <- matrix(rpois(n*p, 2), n, p); Y[sample(length(Y), 30)] <- NA
X <- cbind(1, rnorm(n*p), rnorm(n*p))  # (n*p) x d, vectorized design

fit <- Miss.ZIPLNPCA_Steps(Y, X, q)
fit$elbo
plot(fit$elboPath, type = "l", xlab = "outer iter", ylab = "ELBO")
str(fit$mStep); str(fit$eStep)
} # }
```
