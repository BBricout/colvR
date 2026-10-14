# VectorToMatrix

Convert a numeric vector to a numeric matrix (column-major order).
Implemented in C++ via Rcpp/Armadillo.

## Usage

``` r
VectorToMatrix(vector, n, p)
```

## Arguments

- vector:

  Numeric vector of length n\*p.

- n:

  Integer, number of rows.

- p:

  Integer, number of columns.

## Value

Numeric matrix (n x p).

## See also

[`MatrixToVector`](https://bbricout.github.io/colvR/reference/MatrixToVector.md)
