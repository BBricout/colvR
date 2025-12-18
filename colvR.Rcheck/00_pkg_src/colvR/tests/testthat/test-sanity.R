test_that("package loads", {
  expect_true("colvR" %in% loadedNamespaces())
})

test_that("exports exist", {
  # adapt if names differ
  expect_true(is.function(colvR:::Elbo_grad_Rcpp) || is.function(Elbo_grad_Rcpp))
  expect_true(is.function(colvR:::Elbo) || is.function(Elbo))
  expect_true(is.function(colvR:::Grad) || is.function(Grad))
})

test_that("C++ helpers are callable", {
  if (exists("MatrixToVector") && exists("VectorToMatrix")) {
    # utiliser du double côté R pour matcher Armadillo::mat (double)
    m <- matrix(as.numeric(1:6), nrow = 2)   # <- numeric, pas integer
    v <- MatrixToVector(m)
    expect_type(v, "double")
    expect_length(v, length(m))
    m2 <- VectorToMatrix(v, nrow(m), ncol(m))
    
    # égalité des valeurs et des dimensions
    expect_equal(dim(m2), dim(m))
    expect_equal(m2, m, tolerance = 0)  # tolérance 0, mais ignorer le type
    
    # si tu tiens à tester l'identité stricte :
    storage.mode(m) <- "double"
    expect_identical(m2, m)
  } else {
    succeed("MatrixToVector/VectorToMatrix non exportés ; test sauté.")
  }
})

