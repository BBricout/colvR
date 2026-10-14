# Initialize parameters for the ZIP case (zero-inflated)

SVD-based initialization of parameters for a zero-inflated Poisson
log-normal latent factor model (ZIP-PLN). Regression coefficients for
the abundance part (`B`) are obtained by regressing `log(1+vec(Y))` on
`X`; zero-inflation coefficients (`D`) come from a logistic regression
of `I(Y>0)` on `X`. The latent structure is initialized from the SVD of
the residual matrix.

## Usage

``` r
Init_ZIP(Y, X, q)
```

## Arguments

- Y:

  Numeric `n x p` count matrix (may contain `NA`).

- X:

  Numeric design matrix with `n*p` rows and `d` columns, aligned with
  `vec(Y)` (column-wise vectorization).

- q:

  Integer, target latent dimension (rank).

## Value

A list with elements:

- `B`:

  Abundance (Poisson) regression coefficients (`1 x d`).

- `D`:

  Zero-inflation (logit) regression coefficients (`1 x d`).

- `C`:

  Loadings matrix (`p x q`).

- `M`:

  Variational means of latent factors (`n x q`).

- `S`:

  Variational scale parameters.

## See also

[`Init`](https://bbricout.github.io/colvR/reference/Init.md) for the
non-ZI initializer;
[`MatrixToVector`](https://bbricout.github.io/colvR/reference/MatrixToVector.md),
[`VectorToMatrix`](https://bbricout.github.io/colvR/reference/VectorToMatrix.md)

## Examples

``` r
set.seed(1)
n <- 30; p <- 10; d <- 3; q <- 2
Y <- matrix(rpois(n*p, 2), n, p)
X <- cbind(1, rnorm(n*p), rnorm(n*p))  # (n*p) x d, vectorized design
init <- Init_ZIP(Y, X, q)
str(init)
#> List of 5
#>  $ B: num [1:3, 1] 0.9676 -0.0346 -0.0303
#>   ..- attr(*, "dimnames")=List of 2
#>   .. ..$ : chr [1:3] "X1" "X2" "X3"
#>   .. ..$ : NULL
#>  $ D: num [1:3, 1] 2.01 -0.113 -0.248
#>   ..- attr(*, "dimnames")=List of 2
#>   .. ..$ : chr [1:3] "X1" "X2" "X3"
#>   .. ..$ : NULL
#>  $ C: num [1:10, 1:2] 0.0295 -0.2115 -0.0742 -0.271 -0.2691 ...
#>  $ M: num [1:30, 1:2] -0.09895 -0.11673 0.01751 0.02207 0.00335 ...
#>  $ S: num [1:30, 1:2] 0.333 0.333 0.333 0.333 0.333 ...
```
