# Initialize parameters for a PLN-PCA with missing data

Provides a quick, SVD‑based initialization of model and variational
parameters from a (possibly sparse) count matrix `Y` and a design matrix
`X`. The routine fits a (log‑transformed) linear model to obtain
regression coefficients and uses the SVD of residuals to initialize the
latent structure (loadings and variational means).

## Usage

``` r
Init(Y, X, q)
```

## Arguments

- Y:

  Numeric `n x p` count matrix (may contain `NA`).

- X:

  Numeric design matrix of covariates. Either `n x d` (rowwise design)
  or `(n*p) x d` (vectorized design matching `MatrixToVector(Y)`).

- q:

  Integer, target latent dimension (rank) for the factor structure.

## Value

A list with elements:

- `B`:

  Matrix of regression coefficients (dimensions follow the chosen
  design; typically (`1 x d`) in the rowwise case).

- `C`:

  Loadings matrix (`p x q`) for the latent factors.

- `M`:

  Variational means of the latent factors (`n x q`).

- `S`:

  Variational standard deviations (initialized to a small constant;
  `n x q`).

## Details

Missing values in `Y` are handled via `na.exclude` in the linear model;
residuals at missing entries are then set to zero before the SVD.

## See also

[`MatrixToVector`](https://bbricout.github.io/colvR/reference/MatrixToVector.md),
[`VectorToMatrix`](https://bbricout.github.io/colvR/reference/VectorToMatrix.md)

## Examples

``` r
set.seed(1)
n <- 30; p <- 8; d <- 2; q <- 2
Y <- matrix(rpois(n * p, 2), n, p)
X <- cbind(1, rnorm(n))          # rowwise design
init <- Init(Y, X, q)
str(init)
#> List of 4
#>  $ B: num [1:2, 1:8] 0.9436 -0.105 1.0159 0.0321 1.024 ...
#>   ..- attr(*, "dimnames")=List of 2
#>   .. ..$ : chr [1:2] "X1" "X2"
#>   .. ..$ : NULL
#>  $ C: num [1:8, 1:2] 0.00246 0.20926 0.01447 0.4303 0.09398 ...
#>  $ M: num [1:30, 1:2] -0.4534 0.1827 -0.1048 0.0622 -0.1113 ...
#>  $ S: num [1:30, 1:2] 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 ...
```
