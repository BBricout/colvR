#' MatrixToVector
#'
#' Convert a numeric matrix to a numeric vector (column-major order).
#' Implemented in C++ via Rcpp/Armadillo.
#'
#' @param matrix Numeric matrix (n x p).
#' @return Numeric vector of length n*p.
#' @usage MatrixToVector(matrix)
#' @seealso \code{\link{VectorToMatrix}}
#' @keywords internal
#' @name MatrixToVector
NULL

#' VectorToMatrix
#'
#' Convert a numeric vector to a numeric matrix (column-major order).
#' Implemented in C++ via Rcpp/Armadillo.
#'
#' @param vector Numeric vector of length n*p.
#' @param n Integer, number of rows.
#' @param p Integer, number of columns.
#' @return Numeric matrix (n x p).
#' @usage VectorToMatrix(vector, n, p)
#' @seealso \code{\link{MatrixToVector}}
#' @keywords internal
#' @name VectorToMatrix
NULL
