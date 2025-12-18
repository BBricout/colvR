
#' @export
MatrixToVector <- function(matrix) {
  .Call('_colvR_MatrixToVector', PACKAGE = 'colvR', matrix)
}

#' @export
VectorToMatrix <- function(vector, n, p) {
  .Call('_colvR_VectorToMatrix', PACKAGE = 'colvR', vector, n, p)
}
