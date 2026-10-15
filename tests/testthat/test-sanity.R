test_that("package loads", {
  expect_true("colvR" %in% loadedNamespaces())
})

test_that("exports exist", {
  expect_true(is.function(colvR:::Elbo_grad_Rcpp) || is.function(Elbo_grad_Rcpp))
  expect_true(is.function(colvR:::Elbo) || is.function(Elbo))
  expect_true(is.function(colvR:::Grad) || is.function(Grad))
})

test_that("MatrixToVector and VectorToMatrix are inverse of each other", {
  m <- matrix(as.numeric(1:6), nrow = 2)   # double, to match arma::mat
  v <- MatrixToVector(m)
  expect_type(v, "double")
  expect_length(v, length(m))
  expect_identical(VectorToMatrix(v, nrow(m), ncol(m)), m)
})
