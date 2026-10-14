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
  cl = NULL
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
if (FALSE) { # \dontrun{
pi_cond <- Predictions(Y, X, fit, MC = 100)
pi_marg <- Predictions(Y, X, fit, MC = 100, type = "marginal",
                       probs = c(0.05, 0.95))

# Parallel VE steps on a server
cl <- parallel::makeCluster(8)
res <- Predictions(Y, X, fit, MC = 1000, output = "all", cl = cl)
parallel::stopCluster(cl)
} # }
```
