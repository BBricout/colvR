dat <- sim_data()
fit <- Miss.ZIPLNPCA(dat$Y, dat$X, dat$q)
n <- dat$n; p <- dat$p; d <- dat$d; q <- dat$q
MC <- 5

test_that("default output contains the intervals and the settings", {
  set.seed(1)
  res <- Predictions(dat$Y, dat$X, fit, MC, verbose = FALSE)
  expect_named(res, c("lower", "upper", "settings"))
  expect_equal(dim(res$lower), c(n, p))
  expect_equal(dim(res$upper), c(n, p))
  expect_true(all(res$lower <= res$upper))
  expect_equal(res$settings, list(type = "conditional", probs = c(0, 0.9), MC = MC))
})

test_that("output = 'all' returns draws, latent variables and parameters", {
  set.seed(1)
  res <- Predictions(dat$Y, dat$X, fit, MC, output = "all", verbose = FALSE)
  expect_equal(dim(res$draws$count), c(n, p, MC))
  expect_equal(dim(res$draws$presence), c(n, p, MC))
  expect_equal(dim(res$latent$Z), c(n, p, MC))
  expect_equal(dim(res$latent$M), c(n, q, MC))
  expect_equal(dim(res$latent$S), c(n, q, MC))
  expect_equal(dim(res$latent$xi), c(n, p, MC))
  expect_equal(dim(res$params$gamma), c(MC, d))
  expect_equal(dim(res$params$beta), c(MC, d))
  expect_equal(dim(res$params$C), c(MC, p * q))
})

test_that("the intervals are the quantiles of the draws", {
  set.seed(1)
  res <- Predictions(dat$Y, dat$X, fit, MC, probs = c(0.1, 0.9),
                     output = "draws", verbose = FALSE)
  pred <- res$draws$count * res$draws$presence
  expect_equal(res$lower, apply(pred, c(1, 2), quantile, 0.1, names = FALSE))
  expect_equal(res$upper, apply(pred, c(1, 2), quantile, 0.9, names = FALSE))
})

test_that("marginal intervals work and have no VE outputs", {
  set.seed(1)
  res <- Predictions(dat$Y, dat$X, fit, MC, type = "marginal",
                     output = "latent", verbose = FALSE)
  expect_equal(res$settings$type, "marginal")
  expect_named(res$latent, "Z")
  expect_true(all(res$lower <= res$upper))
})

test_that("results are reproducible with a seed", {
  set.seed(42); r1 <- Predictions(dat$Y, dat$X, fit, MC, verbose = FALSE)
  set.seed(42); r2 <- Predictions(dat$Y, dat$X, fit, MC, verbose = FALSE)
  expect_identical(r1, r2)
})

test_that("parallel VE steps give the same results as sequential ones", {
  skip_on_cran()
  skip_on_os("windows")
  set.seed(1); r_seq <- Predictions(dat$Y, dat$X, fit, MC, output = "all", verbose = FALSE)
  set.seed(1); r_par <- Predictions(dat$Y, dat$X, fit, MC, output = "all", verbose = FALSE, cl = 2)
  expect_identical(r_seq, r_par)
})

test_that("verbose = FALSE prints no message", {
  expect_no_message(Predictions(dat$Y, dat$X, fit, MC, verbose = FALSE))
})

test_that("invalid arguments give errors", {
  expect_error(Predictions(dat$Y, dat$X, fit, MC, probs = c(0.9, 0.1)), "probs")
  expect_error(Predictions(dat$Y, dat$X, fit, MC, probs = 0.5), "probs")
  expect_error(Predictions(dat$Y, dat$X, fit, MC, output = "other"))
  expect_error(Predictions(dat$Y, dat$X, fit, MC, type = "other"))
})
