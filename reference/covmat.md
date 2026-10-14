# Wrapper for \`lori::covmat\`

Thin helper that calls
[`lori::covmat()`](https://rdrr.io/pkg/lori/man/covmat.html) without
requiring users to attach the lori package. It constructs the
block-structured covariate matrix \\X\\ (with \\n \times p\\ rows) from
site-level covariates `R`, year-level covariates `C`, and optional
site–year covariates `E`. Any of `R`, `C`, or `E` can be `NULL` to omit
that block.

## Usage

``` r
covmat(n, p, R = NULL, C = NULL, E = NULL, center = FALSE)
```

## Arguments

- n:

  Integer; number of sites.

- p:

  Integer; number of years.

- R:

  Optional numeric matrix with \\n\\ rows and \\d_R\\ columns
  (site-level covariates). Default: `NULL`.

- C:

  Optional numeric matrix with \\p\\ rows and \\d_C\\ columns
  (year-level covariates). Default: `NULL`.

- E:

  Optional numeric matrix with \\n p\\ rows and \\d_E\\ columns
  (site–year covariates; one row per site–year observation in the same
  order as \\X\\). Default: `NULL`.

- center:

  Logical; whether to center columns within each block (as in
  [`lori::covmat`](https://rdrr.io/pkg/lori/man/covmat.html)). Default:
  `FALSE`.

## Value

A numeric matrix with \\n \times p\\ rows and \\d = d_R + d_C + d_E\\
columns (when present), identical to the value returned by
[`lori::covmat()`](https://rdrr.io/pkg/lori/man/covmat.html).

## See also

[`covmat`](https://rdrr.io/pkg/lori/man/covmat.html)

## Examples

``` r
if (requireNamespace("lori", quietly = TRUE)) {
  set.seed(1)
  n <- 3; p <- 2
  R <- matrix(rnorm(n * 2), nrow = n, ncol = 2)  # site-level (d_R = 2)
  C <- matrix(rnorm(p * 2), nrow = p, ncol = 2)  # year-level (d_C = 2)
  X <- covmat(n, p, R = R, C = C)
  dim(X)  # 6 x 4
}
#> [1] 6 4
```
