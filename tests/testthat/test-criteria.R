n <- 30; p <- 5; d <- 2
Y <- matrix(1, n, p)
X_vec <- matrix(1, n * p, d)        # vectorized design
X_row <- matrix(1, n, d)            # rowwise design

test_that("BIC uses 2d regression parameters for ZIPLNPCA", {
  q <- 2
  fit <- fake_zip_fit(n, p, q)
  expected <- fit$elbo - (q * (p - (q - 1) / 2) + 2 * d) * log(n) / 2
  expect_equal(BIC(Y, X_vec, fit, q), expected)
})

test_that("BIC uses d (vectorized) or p*d (rowwise) parameters for PLNPCA", {
  q <- 2
  fit <- fake_pln_fit(n, q)
  pen_C <- q * (p - (q - 1) / 2)
  expect_equal(BIC(Y, X_vec, fit, q, model = "PLNPCA"),
               fit$elbo - (pen_C + d) * log(n) / 2)
  expect_equal(BIC(Y, X_row, fit, q, model = "PLNPCA"),
               fit$elbo - (pen_C + p * d) * log(n) / 2)
})

test_that("ICL subtracts the entropies of W (S = variance) and U for ZIPLNPCA", {
  q <- 2
  fit <- fake_zip_fit(n, p, q, S = 0.3, xi = 0.7)
  H_W <- 0.5 * n * q * log(0.3) + n * q / 2 * (1 + log(2 * pi))
  H_U <- -n * p * (0.7 * log(0.7) + 0.3 * log(0.3))
  expect_equal(ICL(Y, X_vec, fit, q), BIC(Y, X_vec, fit, q) - H_W - H_U)
})

test_that("ICL uses S^2 (S = standard deviation) for PLNPCA", {
  q <- 2
  fit <- fake_pln_fit(n, q, S = 0.3)
  H_W <- 0.5 * n * q * log(0.3^2) + n * q / 2 * (1 + log(2 * pi))
  expect_equal(ICL(Y, X_vec, fit, q, model = "PLNPCA"),
               BIC(Y, X_vec, fit, q, model = "PLNPCA") - H_W)
})

test_that("ICL is finite when xi contains 0 and 1", {
  fit <- fake_zip_fit(n, p, 2)
  fit$eStep$xi[1, 1] <- 0
  fit$eStep$xi[2, 1] <- 1
  expect_true(is.finite(ICL(Y, X_vec, fit, 2)))
})

test_that("an invalid model gives an error", {
  fit <- fake_zip_fit(n, p, 2)
  expect_error(BIC(Y, X_vec, fit, 2, model = "other"))
  expect_error(ICL(Y, X_vec, fit, 2, model = "other"))
})
