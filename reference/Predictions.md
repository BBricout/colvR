# Predictive simulations and intervals

Draws parameter samples using the sandwich variance `V_theta`, simulates
predictive counts with zero-inflation and returns per-cell prediction
intervals. Two types of intervals are available: conditional (latent
variables drawn from their variational posterior, after one VE step per
draw) and marginal (latent variables drawn from their prior).

## Usage

``` r
Predictions(
  Y,
  X,
  fit,
  MC,
  type = c("conditional", "marginal"),
  probs = c(0, 0.9),
  output = "intervals",
  cl = NULL,
  verbose = TRUE
)
```

## Arguments

- Y:

  An n x p count matrix (NAs allowed).

- X:

  An (n\*p) x d design matrix (stacked by blocks of size n).

- fit:

  Estimator output list (from `Miss.ZIPLNPCA`).

- MC:

  Integer, number of Monte Carlo draws.

- type:

  `"conditional"` (default) or `"marginal"`.

  conditional

  :   for each parameter draw, a VE step gives \\M\\, \\S\\ and \\\xi\\;
      then \\U \sim \mathcal{B}(\xi)\\ and \\Z_i \sim
      \mathcal{N}(X\beta + M_i C^\top, C\\\mathrm{diag}(S_i)\\C^\top)\\.

  marginal

  :   no VE step; \\U \sim \mathcal{B}(\mathrm{logit}^{-1}(X\gamma))\\
      and \\Z_i \sim \mathcal{N}(X\beta, C C^\top)\\.

  In both cases \\U = 1\\ means presence and the prediction is \\U
  \times \mathcal{P}(\exp(Z))\\.

- probs:

  Numeric vector of length 2, the lower and upper quantile levels of the
  interval. Default `c(0, 0.9)`.

- output:

  Character vector, what to return besides the intervals. One or several
  of `"intervals"` (always returned), `"draws"`, `"latent"`, `"params"`,
  or `"all"`.

- cl:

  Parallelisation of the VE steps (conditional type only), passed to
  [`pblapply`](https://peter.solymos.org/pbapply/reference/pbapply.html).
  `NULL` (default): sequential. An integer: number of forked processes
  (not available on Windows). A cluster from
  [`makeCluster`](https://rdrr.io/r/parallel/makeCluster.html): works on
  all platforms. All random draws are made in the main process, so
  results for a given seed do not depend on `cl`.

- verbose:

  Logical. If `TRUE` (default), prints the current step and shows
  progress bars for the VE steps and the simulation of the latent
  variables. Progress bars can also be controlled globally with
  [`pboptions`](https://peter.solymos.org/pbapply/reference/pboptions.html).

## Value

A list with elements:

- `lower`, `upper`: n x p matrices of predictive quantiles at levels
  `probs`.

- `draws` (if requested): list with `count` (Poisson draws) and
  `presence` (Bernoulli draws \\U\\), arrays n x p x MC. The prediction
  is `count * presence`.

- `latent` (if requested): list with `Z`, array n x p x MC of Gaussian
  latent draws, and for the conditional type `M`, `S` (arrays n x q
  x MC) and `xi` (array n x p x MC) from the VE steps.

- `params` (if requested): list with `gamma`, `beta` (matrices MC x d)
  and `C` (matrix MC x (p\*q), column-major vectorisation of C).

- `settings`: list with `type`, `probs` and `MC`.

## See also

[`V_theta`](https://bbricout.github.io/colvR/reference/V_theta.md),
[`Miss.ZIPLNPCA_VE`](https://bbricout.github.io/colvR/reference/Miss.ZIPLNPCA_VE.md)

## Examples

``` r
set.seed(1)
n <- 30; p <- 5; d <- 2; q <- 2
X <- cbind(1, rnorm(n * p))                       # (n*p) x d design
theta <- list(B = matrix(c(1, 0.5)), D = matrix(c(1, -0.5)),
              C = matrix(rnorm(p * q, sd = 0.5), p, q))
Y <- Simul(X, theta, list(n = n, p = p, d = d, q = q))$Y
Y[sample(n * p, 20)] <- NA                        # missing entries
fit <- Miss.ZIPLNPCA(Y, X, q)

# \donttest{
# Conditional intervals (default), 90% upper bound
pi_cond <- Predictions(Y, X, fit, MC = 20)
#> Computing the sandwich variance...
#> Running 20 VE steps:
#> Simulating the latent variables:
#> Simulating the counts and computing the intervals...
pi_cond$upper[1:3, ]
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,] 48.0  7.7 76.8  1.0  1.1
#> [2,]  7.0  7.0  8.0 25.2  5.0
#> [3,]  5.3  0.0 11.0  0.0  5.3

# Marginal intervals at other levels, with the draws
pi_marg <- Predictions(Y, X, fit, MC = 20, type = "marginal",
                       probs = c(0.05, 0.95), output = "draws")
#> Computing the sandwich variance...
#> Simulating the latent variables:
#> Simulating the counts and computing the intervals...
dim(pi_marg$draws$count)
#> [1] 30  5 20
# }
```
