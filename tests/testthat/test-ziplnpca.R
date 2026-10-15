dat <- sim_data()
fit <- Miss.ZIPLNPCA(dat$Y, dat$X, dat$q)
miss <- is.na(dat$Y)

test_that("output dimensions are consistent", {
  expect_equal(dim(fit$mStep$C), c(dat$p, dat$q))
  expect_equal(dim(fit$eStep$M), c(dat$n, dat$q))
  expect_equal(dim(fit$imputed), dim(dat$Y))
  expect_true(is.finite(fit$elbo))
})

test_that("predicted is the ZIP expectation xi * A for all entries", {
  expect_equal(fit$pred$predicted, fit$eStep$xi * fit$pred$A)
})

test_that("imputed equals predicted on missing entries and Y elsewhere", {
  expect_equal(fit$imputed[miss], fit$pred$predicted[miss])
  expect_equal(fit$imputed[!miss], dat$Y[!miss])
  expect_false(anyNA(fit$imputed))
})
