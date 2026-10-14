# PLN-PCA with missing data (variational fit + imputation)

Fits a Poisson log-normal PCA model (PLN-PCA) to a count matrix with
missing values, using a variational objective optimized by NLOpt-based
routines. The function estimates model/variational parameters and
returns expected counts to impute missing entries.

## Usage

``` r
Miss.PLNPCA(Y, X, O = NULL, w = NULL, q, params = NULL, config = NULL)
```

## Arguments

- Y:

  Numeric `n x p` count matrix. May contain `NA`.

- X:

  Numeric design matrix of covariates: either `n x d` (rowwise) or
  `(n*p) x d` (vectorized to match `MatrixToVector(Y)`).

- O:

  Optional numeric `n x p` matrix of offsets (default: zeros).

- w:

  Optional numeric vector of length `n` with observation weights
  (default: all ones).

- q:

  Integer, latent rank (dimension of the latent space).

- params:

  Optional initial parameters (list) typically produced by
  [`Init`](https://bbricout.github.io/colvR/reference/Init.md); if
  `NULL`, they are initialized internally.

- config:

  Optional optimizer configuration; if `NULL`, defaults to
  `PLNPCA_param()$config_optim`.

## Value

A list with components:

- `mStep`:

  List with `beta` (`1 x d`) and `C` (`p x q`).

- `eStep`:

  List with `M` (`n x q`) and `S` (`n x q`).

- `pred`:

  List with `A` and `predicted`, both equal to the `n x p` matrix of
  expected counts used for imputation: \$\$ A = \exp\\\big(O + X B + M
  C^\top + 0.5\\(S\odot S)\\(C\odot C)^\top\big), \$\$ where \\\odot\\
  denotes the Hadamard product and `X B` is reshaped to `n x p` when `X`
  is vectorized.

- `iter`:

  Integer, number of iterations.

- `elboPath`:

  Numeric vector of objective (ELBO) values over iterations.

- `elbo`:

  Final ELBO value.

- `params.init`:

  The initial parameters used.

- `monitoring`:

  Optimizer log/diagnostics as returned by the backend.

## Details

Two designs for `X` are supported:

1.  **Rowwise design** (`nrow(X) == n`): one row per sample.

2.  **Vectorized design** (`nrow(X) == n*p`): `X` aligns with `vec(Y)`
    (vectorization by columns).

## See also

[`Init`](https://bbricout.github.io/colvR/reference/Init.md),
[`PLNPCA_param`](https://bbricout.github.io/colvR/reference/PLNPCA_param.md),
[`MatrixToVector`](https://bbricout.github.io/colvR/reference/MatrixToVector.md),
[`VectorToMatrix`](https://bbricout.github.io/colvR/reference/VectorToMatrix.md),
[`Elbo_grad`](https://bbricout.github.io/colvR/reference/Elbo_grad.md)

## Examples

``` r
if (FALSE) { # \dontrun{
set.seed(1)
n <- 40; p <- 12; d <- 2; q <- 2
Y <- matrix(rpois(n*p, 2), n, p)
Y[sample(length(Y), 20)] <- NA
X <- cbind(1, rnorm(n))      # rowwise design

fit <- Miss.PLNPCA(
  Y = Y, X = X, q = q,
  O = matrix(0, n, p),
  w = rep(1, n),
  params = NULL,
  config = NULL
)
str(fit$pred$A)     # expected counts (n x p)
fit$elbo            # final ELBO
} # }
```
