n <- 30; p <- 5; d <- 2
Y <- matrix(1, n, p)
X <- matrix(1, n * p, d)

# Fake fitting functions: the ELBO is maximal (after penalty) around q = 4
fake_fun_zip <- function(Y, X, q) fake_zip_fit(nrow(Y), ncol(Y), q, elbo = -1000 + 300 * q - 40 * q^2)
fake_fun_pln <- function(Y, X, q) fake_pln_fit(nrow(Y), q, elbo = -1000 + 300 * q - 40 * q^2)

test_that("output has the expected structure", {
  sel <- select_model_bic(Y, X, fake_fun_zip, latents = 1:3)
  expect_named(sel, c("criterion", "values", "q_star", "fit_star"))
  expect_equal(sel$criterion, "BIC")
  expect_length(sel$values, 3)
  expect_named(sel$values, c("1", "2", "3"))
})

test_that("q_star maximizes the criterion and fit_star is fitted with q_star", {
  sel <- select_model_bic(Y, X, fake_fun_zip, latents = 1:6)
  expect_equal(sel$q_star, as.numeric(names(which.max(sel$values))))
  expect_equal(sel$fit_star$q, sel$q_star)
  expect_equal(unname(sel$values[as.character(sel$q_star)]),
               BIC(Y, X, sel$fit_star, sel$q_star))
})

test_that("a non-contiguous grid of latent dimensions works", {
  sel <- select_model_bic(Y, X, fake_fun_zip, latents = c(2, 4, 6))
  expect_named(sel$values, c("2", "4", "6"))
  expect_true(sel$q_star %in% c(2, 4, 6))
  expect_equal(sel$fit_star$q, sel$q_star)
})

test_that("criterion = 'ICL' uses ICL", {
  sel <- select_model_bic(Y, X, fake_fun_zip, latents = 1:3, criterion = "ICL")
  expect_equal(sel$criterion, "ICL")
  expect_equal(unname(sel$values["2"]), ICL(Y, X, fake_fun_zip(Y, X, 2), 2))
})

test_that("model = 'PLNPCA' works with PLNPCA fits", {
  sel <- select_model_bic(Y, X, fake_fun_pln, latents = 1:3, model = "PLNPCA")
  expect_null(sel$fit_star$mStep$gamma)
})

test_that("a mismatch between fun and model stops after the first fit", {
  n_calls <- 0
  counting_fun <- function(Y, X, q) { n_calls <<- n_calls + 1; fake_fun_pln(Y, X, q) }
  expect_error(select_model_bic(Y, X, counting_fun, latents = 1:5), "PLNPCA")
  expect_equal(n_calls, 1)
  expect_error(select_model_bic(Y, X, fake_fun_zip, latents = 1:2, model = "PLNPCA"),
               "ZIPLNPCA")
})

test_that("select_model_bic works with Miss.ZIPLNPCA", {
  skip_on_cran()
  dat <- sim_data()
  sel <- select_model_bic(dat$Y, dat$X, Miss.ZIPLNPCA, latents = 1:2)
  expect_true(sel$q_star %in% 1:2)
  expect_equal(ncol(sel$fit_star$mStep$C), sel$q_star)
})
