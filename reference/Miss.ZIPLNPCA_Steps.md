# ZI‑PLN-PCA (missing data) with parameterwise optimization steps

Fits a zero‑inflated Poisson log‑normal (ZI‑PLN-PCA) latent factor model
to a count matrix with missing values using a **blockwise** optimization
strategy. At each outer iteration, parameters are updated by activating
one block among {B, D, C, M, S} while keeping the others fixed, via an
NLOpt backend. The loop stops when both parameter and ELBO changes are
below thresholds or when `maxiter` is reached.

## Usage

``` r
Miss.ZIPLNPCA_Steps(Y, X, q, params = NULL, config_vem = NULL, config = NULL)
```

## Arguments

- Y:

  Numeric `n x p` count matrix. May contain `NA`.

- X:

  Numeric design matrix with `n*p` rows and `d` columns, aligned with
  `vec(Y)` (column‑wise vectorization).

- q:

  Integer, latent rank (dimension of the latent space).

- params:

  Optional list of initial parameters. If `NULL`,
  [`Init_ZIP`](https://bbricout.github.io/colvR/reference/Init_ZIP.md)
  is used.

- config_vem:

  List of controls for the outer blockwise loop (variational EM‑like),
  with fields:

  `maxiter`

  :   Maximum number of outer iterations (default `1e6`).

  `ftol`

  :   ELBO tolerance; stop if `|ELBO_{t}-ELBO_{t-1}| <= ftol`.

  `xtol`

  :   Parameter tolerance; stop if `max|theta_{t}-theta_{t-1}| <= xtol`.

  `tolS`

  :   List with `lower`, `upper` bounds for `S`.

  `tolxi`

  :   Tolerance used in the Jaakkola‑type \\\xi\\ updates for the
      logistic bound.

- config:

  List of controls for the NLOpt backend (per‑block inner solve), e.g.
  `algorithm`, `maxeval`, `ftol_abs`, `xtol_abs`, `ftol_rel`,
  `xtol_rel`, `trace`, `maxtime`, as well as `upper_bounds`,
  `lower_bounds` (filled here from `tolS` and dimensions).

## Value

A list with components:

- `mStep`:

  Model parameters: `gamma` (`d x 1`), `beta` (`d x 1`), loadings `C`
  (`p x q`).

- `eStep`:

  Variational parameters: `M` (`n x q`), `S` (`n x q`), logistic bound
  parameters `xi` (`n x p`).

- `pred`:

  List with predictors and expected counts: `mu` (`n x p`), `nu`
  (`n x p`), backend mean `A` (`n x p`), and `predicted` (`n x p`, ZIP
  expectation for all entries: \\\xi \odot A\\, with \\A = \exp( XB + M
  C^\top + 0.5\\ S\\ (C\odot C)^\top )\\, `S` being the variational
  variances).

- `imputed`:

  `n x p` matrix equal to `predicted` at missing entries of `Y`, and `Y`
  elsewhere.

- `iter`:

  Number of outer iterations performed.

- `elboPath`:

  Numeric vector of ELBO values across outer iterations.

- `elbo`:

  Final ELBO value.

- `params.init`:

  Parameters used for initialization.

- `monitoring`:

  List with `status` (stopping reason) and `iterations`.

- `gradB, gradD, gradC, gradM, gradS`:

  ELBO gradients w.r.t. each parameter block (from
  [`Elbo_grad`](https://bbricout.github.io/colvR/reference/Elbo_grad.md))
  at the solution.

## Details

Missing entries are handled through a binary mask and a working copy of
`Y` with zeros at missing locations for the objective evaluation.

A binary mask `R = 1_{observed}(Y)` is built; a working matrix `Y.na`
replaces missings by 0 for the objective. Box constraints for `S` are
set using `config_vem$tolS` and injected into `config$upper_bounds` and
`config$lower_bounds`. Each block update calls the backend
`nlopt_optimize_ZIP_Steps(data, params, config, tolxi, active_blocks)`
with a single active block at a time.

**Predicted mean:** in this implementation, the Poisson mean used for
the R‑side recomputation is \$\$ \exp\\\big( \mu + M C^\top + 0.5\\ S
\\(C\odot C)^\top \big), \$\$ where \\\mu = X B\\ reshaped into `n x p`.
Note that other functions in le package emploient \\0.5\\(S\odot
S)\\(C\odot C)^\top\\. Vérifie que la paramétrisation voulue est
cohérente avec le backend.

## See also

[`Miss.ZIPLNPCA`](https://bbricout.github.io/colvR/reference/Miss.ZIPLNPCA.md),
[`Miss.ZIPLNPCA.logS`](https://bbricout.github.io/colvR/reference/Miss.ZIPLNPCA.logS.md),
[`Init_ZIP`](https://bbricout.github.io/colvR/reference/Init_ZIP.md),
[`Elbo_grad`](https://bbricout.github.io/colvR/reference/Elbo_grad.md)

## Examples

``` r
set.seed(1)
n <- 30; p <- 5; d <- 2; q <- 2
X <- cbind(1, rnorm(n * p))                       # (n*p) x d design
theta <- list(B = matrix(c(1, 0.5)), D = matrix(c(1, -0.5)),
              C = matrix(rnorm(p * q, sd = 0.5), p, q))
Y <- Simul(X, theta, list(n = n, p = p, d = d, q = q))$Y
Y[sample(n * p, 20)] <- NA                        # missing entries

fit <- Miss.ZIPLNPCA_Steps(Y, X, q)
#> Iteration: 1 ELBO: -427.0734 Diff params: 1 Diff ELBO: 1 
#> gradB : 9.2473e-11 gradD : 3.3309e+00 gradC : 2.5051e+02 gradM : 2.0184e+01 gradS : 7.4695e-01
#> gradB : 4.4087e-01 gradD : 1.0667e-09 gradC : 2.5134e+02 gradM : 2.0188e+01 gradS : 7.4697e-01
#> gradB : 1.7220e+02 gradD : 4.8240e-01 gradC : 7.7188e-12 gradM : 1.3758e+00 gradS : 3.5999e-01
#> gradB : 2.8717e+01 gradD : 3.2126e-01 gradC : 3.1037e+03 gradM : 1.1177e-12 gradS : 3.5647e-01
#> gradB : 2.4937e+01 gradD : 2.8663e-01 gradC : 2.9082e+03 gradM : 2.1721e-05 gradS : 1.4755e-10
#> Iteration: 2 ELBO: -311.2045 Diff params: 5.84827 Diff ELBO: 115.869 
#> gradB : 3.6693e-10 gradD : 3.3804e-01 gradC : 2.8867e+03 gradM : 5.0688e-04 gradS : 5.5102e-06
#> gradB : 9.3653e-02 gradD : 5.7951e-13 gradC : 2.8860e+03 gradM : 5.1377e-04 gradS : 5.5775e-06
#> gradB : 6.3205e+02 gradD : 2.4268e-02 gradC : 8.7514e-12 gradM : 1.0269e+00 gradS : 1.6166e-01
#> gradB : 3.8617e+02 gradD : 1.6873e-03 gradC : 4.7859e+02 gradM : 1.1106e-06 gradS : 1.0469e-01
#> gradB : 2.9349e+02 gradD : 9.0327e-04 gradC : 7.5015e+02 gradM : 3.1213e-03 gradS : 1.1337e-07
#> Iteration: 3 ELBO: -274.8792 Diff params: 2.429351 Diff ELBO: 36.32525 
#> gradB : 1.8266e-12 gradD : 4.0692e-01 gradC : 1.0171e+03 gradM : 4.7241e-02 gradS : 1.0211e-03
#> gradB : 1.4698e-01 gradD : 1.4131e-13 gradC : 1.0164e+03 gradM : 4.7266e-02 gradS : 1.0220e-03
#> gradB : 6.5055e+02 gradD : 2.4497e-02 gradC : 6.9516e-10 gradM : 4.0904e-01 gradS : 2.0177e-01
#> gradB : 2.9032e+02 gradD : 8.3732e-03 gradC : 1.5324e+02 gradM : 9.9660e-12 gradS : 1.2231e-01
#> gradB : 2.1858e+02 gradD : 9.3867e-03 gradC : 2.2771e+02 gradM : 1.6919e-03 gradS : 2.1669e-07
#> Iteration: 4 ELBO: -259.4257 Diff params: 0.4867441 Diff ELBO: 15.45349 
#> gradB : 9.8862e-12 gradD : 4.6482e-01 gradC : 3.8639e+02 gradM : 6.1185e-02 gradS : 2.5476e-03
#> gradB : 1.9106e-01 gradD : 3.6296e-14 gradC : 3.8584e+02 gradM : 6.1278e-02 gradS : 2.5438e-03
#> gradB : 3.1075e+02 gradD : 9.3128e-03 gradC : 5.1501e-12 gradM : 2.8653e-01 gradS : 1.6089e-01
#> gradB : 1.5732e+02 gradD : 1.1491e-03 gradC : 7.7778e+01 gradM : 2.6022e-13 gradS : 8.6892e-02
#> gradB : 1.2625e+02 gradD : 4.7860e-03 gradC : 1.0636e+02 gradM : 6.9855e-04 gradS : 5.4863e-10
#> Iteration: 5 ELBO: -253.7549 Diff params: 0.2969835 Diff ELBO: 5.670844 
#> gradB : 5.1118e-13 gradD : 3.7299e-01 gradC : 1.9534e+02 gradM : 5.2094e-02 gradS : 3.1355e-03
#> gradB : 2.0003e-01 gradD : 1.2987e-14 gradC : 1.9495e+02 gradM : 5.2163e-02 gradS : 3.1293e-03
#> gradB : 1.7884e+02 gradD : 4.8708e-03 gradC : 8.8678e-11 gradM : 2.0443e-01 gradS : 1.2668e-01
#> gradB : 9.2674e+01 gradD : 6.4634e-04 gradC : 4.4389e+01 gradM : 3.0575e-07 gradS : 6.2818e-02
#> gradB : 7.6238e+01 gradD : 4.3514e-03 gradC : 5.8205e+01 gradM : 3.9358e-04 gradS : 4.7513e-06
#> Iteration: 6 ELBO: -250.7755 Diff params: 0.2103298 Diff ELBO: 2.979367 
#> gradB : 8.5961e-09 gradD : 2.6613e-01 gradC : 1.1047e+02 gradM : 4.0603e-02 gradS : 3.0819e-03
#> gradB : 1.7056e-01 gradD : 3.6247e-13 gradC : 1.1021e+02 gradM : 4.0656e-02 gradS : 3.0728e-03
#> gradB : 1.1141e+02 gradD : 2.7756e-03 gradC : 3.0329e-10 gradM : 1.5041e-01 gradS : 1.0019e-01
#> gradB : 5.8836e+01 gradD : 9.9239e-04 gradC : 2.7404e+01 gradM : 1.2993e-07 gradS : 4.6660e-02
#> gradB : 4.9292e+01 gradD : 4.6945e-03 gradC : 3.5039e+01 gradM : 2.4173e-04 gradS : 4.4957e-04
#> Iteration: 7 ELBO: -249.0061 Diff params: 0.1591037 Diff ELBO: 1.769366 
#> gradB : 5.4961e-09 gradD : 1.9596e-01 gradC : 6.7747e+01 gradM : 3.1180e-02 gradS : 2.8731e-03
#> gradB : 1.4415e-01 gradD : 6.8656e-13 gradC : 6.7568e+01 gradM : 3.1221e-02 gradS : 2.8641e-03
#> gradB : 7.3140e+01 gradD : 1.5857e-03 gradC : 1.8613e-10 gradM : 1.1329e-01 gradS : 8.1228e-02
#> gradB : 3.9202e+01 gradD : 1.4308e-03 gradC : 1.7927e+01 gradM : 1.5315e-07 gradS : 3.6726e-02
#> gradB : 3.2723e+01 gradD : 4.4534e-03 gradC : 2.2639e+01 gradM : 1.7826e-04 gradS : 4.6658e-06
#> Iteration: 8 ELBO: -247.8691 Diff params: 0.1251954 Diff ELBO: 1.137067 
#> gradB : 7.5583e-13 gradD : 1.4245e-01 gradC : 4.3959e+01 gradM : 2.3792e-02 gradS : 2.4180e-03
#> gradB : 1.1727e-01 gradD : 2.0422e-13 gradC : 4.3834e+01 gradM : 2.3829e-02 gradS : 2.4124e-03
#> gradB : 5.0670e+01 gradD : 9.5142e-04 gradC : 7.4093e-11 gradM : 8.7730e-02 gradS : 6.4737e-02
#> gradB : 2.7219e+01 gradD : 1.7318e-03 gradC : 1.2284e+01 gradM : 3.3614e-09 gradS : 2.7325e-02
#> gradB : 2.3030e+01 gradD : 5.3144e-03 gradC : 1.5173e+01 gradM : 1.0665e-04 gradS : 1.0151e-03
#> Iteration: 9 ELBO: -247.0994 Diff params: 0.1020559 Diff ELBO: 0.769679 
#> gradB : 3.2672e-13 gradD : 1.1335e-01 gradC : 2.9742e+01 gradM : 1.8238e-02 gradS : 1.1387e-03
#> gradB : 1.0253e-01 gradD : 1.1885e-13 gradC : 2.9652e+01 gradM : 1.8271e-02 gradS : 1.1302e-03
#> gradB : 3.6081e+01 gradD : 5.7213e-04 gradC : 7.9004e-11 gradM : 6.8581e-02 gradS : 6.4671e-02
#> gradB : 1.9364e+01 gradD : 2.1477e-03 gradC : 8.6953e+00 gradM : 2.1551e-10 gradS : 2.8134e-02
#> gradB : 1.6378e+01 gradD : 4.7098e-03 gradC : 1.0903e+01 gradM : 8.9397e-05 gradS : 6.1082e-06
#> Iteration: 10 ELBO: -246.5585 Diff params: 0.08458054 Diff ELBO: 0.5408944 
#> gradB : 2.6396e-13 gradD : 8.5229e-02 gradC : 2.1148e+01 gradM : 1.4353e-02 gradS : 1.7231e-03
#> gradB : 8.3319e-02 gradD : 9.7574e-12 gradC : 2.1084e+01 gradM : 1.4381e-02 gradS : 1.7190e-03
#> gradB : 2.6262e+01 gradD : 3.5066e-04 gradC : 3.0071e-11 gradM : 5.4816e-02 gradS : 4.3721e-02
#> gradB : 1.4030e+01 gradD : 1.9724e-03 gradC : 6.3802e+00 gradM : 8.5832e-11 gradS : 1.7292e-02
#> gradB : 1.1808e+01 gradD : 4.5825e-03 gradC : 7.9780e+00 gradM : 6.7269e-05 gradS : 2.6332e-04
#> Iteration: 11 ELBO: -246.1644 Diff params: 0.07161523 Diff ELBO: 0.3940859 
#> gradB : 4.2923e-13 gradD : 6.6573e-02 gradC : 1.5331e+01 gradM : 1.1049e-02 gradS : 2.5935e-03
#> gradB : 6.9712e-02 gradD : 4.5140e-12 gradC : 1.5284e+01 gradM : 1.1073e-02 gradS : 2.5885e-03
#> gradB : 1.9646e+01 gradD : 2.2526e-04 gradC : 5.6961e-13 gradM : 4.4798e-02 gradS : 3.1525e-02
#> gradB : 1.0367e+01 gradD : 1.8672e-03 gradC : 4.8335e+00 gradM : 4.1796e-13 gradS : 1.1367e-02
#> gradB : 8.9851e+00 gradD : 5.4517e-03 gradC : 5.9350e+00 gradM : 4.3893e-05 gradS : 6.2076e-04
#> Iteration: 12 ELBO: -245.8709 Diff params: 0.0617749 Diff ELBO: 0.2934994 
#> gradB : 4.7608e-13 gradD : 5.7221e-02 gradC : 1.1376e+01 gradM : 8.6554e-03 gradS : 1.8319e-03
#> gradB : 6.3496e-02 gradD : 3.8782e-13 gradC : 1.1342e+01 gradM : 8.6783e-03 gradS : 1.8166e-03
#> gradB : 1.4787e+01 gradD : 1.4374e-04 gradC : 7.9697e-10 gradM : 3.6847e-02 gradS : 2.9996e-02
#> gradB : 7.7723e+00 gradD : 1.9032e-03 gradC : 3.7229e+00 gradM : 7.5876e-11 gradS : 1.1347e-02
#> gradB : 6.5021e+00 gradD : 2.4763e-03 gradC : 4.4924e+00 gradM : 4.7962e-05 gradS : 4.2624e-06
#> Iteration: 13 ELBO: -245.65 Diff params: 0.05382094 Diff ELBO: 0.2208847 
#> gradB : 1.4368e-13 gradD : 3.6738e-02 gradC : 8.5087e+00 gradM : 6.8084e-03 gradS : 1.0513e-03
#> gradB : 4.2936e-02 gradD : 2.3462e-12 gradC : 8.4848e+00 gradM : 6.8231e-03 gradS : 1.0499e-03
#> gradB : 1.1273e+01 gradD : 9.3547e-05 gradC : 5.1644e-11 gradM : 3.0061e-02 gradS : 2.5084e-02
#> gradB : 5.8236e+00 gradD : 1.1265e-03 gradC : 2.8593e+00 gradM : 6.0177e-08 gradS : 8.9645e-03
#> gradB : 4.8420e+00 gradD : 3.1410e-03 gradC : 3.6049e+00 gradM : 4.3118e-05 gradS : 9.1117e-04
#> Iteration: 14 ELBO: -245.4798 Diff params: 0.04672583 Diff ELBO: 0.1702003 
#> gradB : 5.2296e-13 gradD : 3.1609e-02 gradC : 6.6482e+00 gradM : 5.2834e-03 gradS : 2.0104e-03
#> gradB : 3.8734e-02 gradD : 2.0543e-12 gradC : 6.6298e+00 gradM : 5.2975e-03 gradS : 2.0007e-03
#> gradB : 8.9956e+00 gradD : 6.9859e-05 gradC : 8.7452e-12 gradM : 2.5795e-02 gradS : 2.0496e-02
#> gradB : 4.4932e+00 gradD : 1.2572e-03 gradC : 2.3179e+00 gradM : 4.2011e-08 gradS : 6.5821e-03
#> gradB : 3.8630e+00 gradD : 2.2878e-03 gradC : 2.8135e+00 gradM : 2.5942e-05 gradS : 2.4252e-03
#> Iteration: 15 ELBO: -245.347 Diff params: 0.04196144 Diff ELBO: 0.1328526 
#> gradB : 3.4920e-13 gradD : 2.3981e-02 gradC : 5.1629e+00 gradM : 4.3327e-03 gradS : 5.0540e-03
#> gradB : 3.0156e-02 gradD : 3.3307e-10 gradC : 5.1495e+00 gradM : 4.3440e-03 gradS : 5.0486e-03
#> gradB : 6.9350e+00 gradD : 4.5468e-05 gradC : 1.3641e-11 gradM : 2.1262e-02 gradS : 8.9233e-03
#> gradB : 3.4451e+00 gradD : 9.3932e-04 gradC : 1.8321e+00 gradM : 5.1834e-08 gradS : 2.4932e-03
#> gradB : 2.9419e+00 gradD : 1.6722e-03 gradC : 2.1465e+00 gradM : 1.5642e-05 gradS : 3.1056e-06
#> Iteration: 16 ELBO: -245.2423 Diff params: 0.03695544 Diff ELBO: 0.1047279 
#> gradB : 1.6244e-13 gradD : 1.8149e-02 gradC : 3.9340e+00 gradM : 3.2475e-03 gradS : 5.2004e-04
#> gradB : 2.3756e-02 gradD : 2.1500e-10 gradC : 3.9242e+00 gradM : 3.2562e-03 gradS : 5.1828e-04
#> gradB : 5.3957e+00 gradD : 3.0974e-05 gradC : 7.2828e-11 gradM : 1.7743e-02 gradS : 1.5491e-02
#> gradB : 2.6565e+00 gradD : 7.2747e-04 gradC : 1.4573e+00 gradM : 6.6588e-08 gradS : 5.1969e-03
#> gradB : 2.2454e+00 gradD : 1.5950e-03 gradC : 1.7528e+00 gradM : 1.5250e-05 gradS : 2.6269e-04
#> Iteration: 17 ELBO: -245.1591 Diff params: 0.03289989 Diff ELBO: 0.08315628 
#> gradB : 1.1494e-09 gradD : 1.4757e-02 gradC : 3.1432e+00 gradM : 2.6285e-03 gradS : 3.6086e-04
#> gradB : 1.9819e-02 gradD : 2.9461e-11 gradC : 3.1357e+00 gradM : 2.6361e-03 gradS : 3.5885e-04
#> gradB : 4.3415e+00 gradD : 2.3124e-05 gradC : 3.1151e-13 gradM : 1.4982e-02 gradS : 1.6063e-02
#> gradB : 2.0843e+00 gradD : 6.4893e-04 gradC : 1.1858e+00 gradM : 7.2519e-08 gradS : 5.7930e-03
#> gradB : 1.7353e+00 gradD : 1.9449e-03 gradC : 1.4911e+00 gradM : 2.2790e-05 gradS : 7.9243e-04
#> Iteration: 18 ELBO: -245.0917 Diff params: 0.02954853 Diff ELBO: 0.06736591 
#> gradB : 8.5007e-10 gradD : 1.3256e-02 gradC : 2.5933e+00 gradM : 2.1306e-03 gradS : 1.3298e-03
#> gradB : 1.8156e-02 gradD : 5.4888e-12 gradC : 2.5874e+00 gradM : 2.1383e-03 gradS : 1.3222e-03
#> gradB : 3.5715e+00 gradD : 1.8372e-05 gradC : 2.1651e-11 gradM : 1.3185e-02 gradS : 1.1941e-02
#> gradB : 1.6642e+00 gradD : 6.4584e-04 gradC : 1.0041e+00 gradM : 7.2222e-08 gradS : 4.2011e-03
#> gradB : 1.4507e+00 gradD : 7.2277e-04 gradC : 1.1608e+00 gradM : 1.2992e-05 gradS : 8.0410e-06
#> Iteration: 19 ELBO: -245.0377 Diff params: 0.02709853 Diff ELBO: 0.05406672 
#> gradB : 7.0118e-10 gradD : 8.4725e-03 gradC : 2.0504e+00 gradM : 1.7668e-03 gradS : 2.7344e-04
#> gradB : 1.1894e-02 gradD : 7.3472e-11 gradC : 2.0463e+00 gradM : 1.7710e-03 gradS : 2.7302e-04
#> gradB : 2.7785e+00 gradD : 1.0719e-05 gradC : 1.5791e-11 gradM : 1.0749e-02 gradS : 1.0033e-02
#> gradB : 1.2936e+00 gradD : 3.5897e-04 gradC : 7.9933e-01 gradM : 1.6026e-07 gradS : 3.2733e-03
#> gradB : 1.0598e+00 gradD : 7.4878e-04 gradC : 9.7694e-01 gradM : 1.0417e-05 gradS : 6.3021e-05
#> Iteration: 20 ELBO: -244.9934 Diff params: 0.02399075 Diff ELBO: 0.04422684 
#> gradB : 4.9280e-10 gradD : 6.8366e-03 gradC : 1.6610e+00 gradM : 1.3476e-03 gradS : 3.4811e-04
#> gradB : 9.8788e-03 gradD : 2.0500e-13 gradC : 1.6578e+00 gradM : 1.3515e-03 gradS : 3.4707e-04
#> gradB : 2.2784e+00 gradD : 9.4882e-06 gradC : 1.1776e-12 gradM : 9.3694e-03 gradS : 8.5182e-03
#> gradB : 1.0274e+00 gradD : 3.4493e-04 gradC : 6.7329e-01 gradM : 1.4244e-07 gradS : 2.6536e-03
#> gradB : 8.6830e-01 gradD : 8.4350e-04 gradC : 8.0522e-01 gradM : 1.0171e-05 gradS : 1.2496e-03
#> Iteration: 21 ELBO: -244.957 Diff params: 0.02202948 Diff ELBO: 0.03638475 
#> gradB : 3.9203e-10 gradD : 6.2085e-03 gradC : 1.3637e+00 gradM : 1.1109e-03 gradS : 1.4801e-03
#> gradB : 8.9643e-03 gradD : 2.7343e-11 gradC : 1.3611e+00 gradM : 1.1145e-03 gradS : 1.4789e-03
#> gradB : 1.8954e+00 gradD : 6.9716e-06 gradC : 8.0483e-12 gradM : 7.9859e-03 gradS : 8.5850e-03
#> gradB : 8.2496e-01 gradD : 3.0097e-04 gradC : 5.5991e-01 gradM : 1.9020e-07 gradS : 3.4398e-03
#> gradB : 6.9079e-01 gradD : 4.9254e-04 gradC : 6.7676e-01 gradM : 1.1013e-05 gradS : 1.1303e-05
#> Iteration: 22 ELBO: -244.927 Diff params: 0.02002149 Diff ELBO: 0.03002192 
#> gradB : 3.0428e-10 gradD : 4.4290e-03 gradC : 1.1249e+00 gradM : 8.9878e-04 gradS : 1.5833e-04
#> gradB : 6.6182e-03 gradD : 1.9015e-10 gradC : 1.1230e+00 gradM : 9.0161e-04 gradS : 1.5753e-04
#> gradB : 1.5118e+00 gradD : 5.3829e-06 gradC : 2.9754e-13 gradM : 6.9240e-03 gradS : 6.5349e-03
#> gradB : 6.6566e-01 gradD : 2.5928e-04 gradC : 4.7096e-01 gradM : 1.4819e-07 gradS : 2.0702e-03
#> gradB : 5.3807e-01 gradD : 6.9662e-04 gradC : 5.9345e-01 gradM : 6.9464e-06 gradS : 2.0357e-04
#> Iteration: 23 ELBO: -244.9013 Diff params: 0.01832623 Diff ELBO: 0.02570174 
#> gradB : 2.2890e-10 gradD : 4.2577e-03 gradC : 9.5840e-01 gradM : 7.2068e-04 gradS : 6.0552e-04
#> gradB : 6.3442e-03 gradD : 1.4227e-10 gradC : 9.5683e-01 gradM : 7.2312e-04 gradS : 6.0263e-04
#> gradB : 1.2967e+00 gradD : 5.0914e-06 gradC : 5.4659e-12 gradM : 6.2004e-03 gradS : 3.9604e-03
#> gradB : 5.3351e-01 gradD : 1.9832e-04 gradC : 4.1417e-01 gradM : 1.7153e-07 gradS : 9.3967e-04
#> gradB : 4.6346e-01 gradD : 3.0759e-04 gradC : 4.7149e-01 gradM : 3.2394e-06 gradS : 5.4871e-05
#> Iteration: 24 ELBO: -244.8807 Diff params: 0.01712729 Diff ELBO: 0.02065397 
#> gradB : 1.9771e-10 gradD : 2.8891e-03 gradC : 7.7468e-01 gradM : 6.0606e-04 gradS : 1.4516e-04
#> gradB : 4.4093e-03 gradD : 7.9700e-11 gradC : 7.7353e-01 gradM : 6.0802e-04 gradS : 1.4410e-04
#> gradB : 1.0386e+00 gradD : 3.2131e-06 gradC : 3.1220e-12 gradM : 5.1345e-03 gradS : 5.0902e-03
#> gradB : 4.2722e-01 gradD : 1.5626e-04 gradC : 3.3708e-01 gradM : 1.3113e-07 gradS : 1.6799e-03
#> gradB : 3.5305e-01 gradD : 2.7374e-04 gradC : 4.0812e-01 gradM : 5.2690e-06 gradS : 1.2493e-08
#> Iteration: 25 ELBO: -244.8631 Diff params: 0.01538339 Diff ELBO: 0.01759311 
#> gradB : 1.4959e-10 gradD : 2.3169e-03 gradC : 6.5118e-01 gradM : 4.8996e-04 gradS : 8.9580e-05
#> gradB : 3.5564e-03 gradD : 1.0166e-12 gradC : 6.5028e-01 gradM : 4.9160e-04 gradS : 8.9246e-05
#> gradB : 8.6283e-01 gradD : 2.2867e-06 gradC : 7.2761e-10 gradM : 4.4869e-03 gradS : 4.2110e-03
#> gradB : 3.4785e-01 gradD : 1.3801e-04 gradC : 2.8821e-01 gradM : 9.8259e-08 gradS : 1.2494e-03
#> gradB : 3.0051e-01 gradD : 3.7912e-04 gradC : 3.4645e-01 gradM : 4.9355e-06 gradS : 4.4886e-04
#> Iteration: 26 ELBO: -244.8479 Diff params: 0.01422532 Diff ELBO: 0.01521129 
#> gradB : 1.2146e-10 gradD : 2.3359e-03 gradC : 5.4838e-01 gradM : 4.0155e-04 gradS : 4.4792e-04
#> gradB : 3.5941e-03 gradD : 8.0153e-12 gradC : 5.4764e-01 gradM : 4.0315e-04 gradS : 4.4688e-04
#> gradB : 7.1251e-01 gradD : 1.9134e-06 gradC : 1.6499e-10 gradM : 3.9059e-03 gradS : 4.1833e-03
#> gradB : 2.8787e-01 gradD : 1.3571e-04 gradC : 2.4507e-01 gradM : 1.2021e-07 gradS : 1.5920e-03
#> gradB : 2.3251e-01 gradD : 1.6686e-04 gradC : 2.9081e-01 gradM : 4.7870e-06 gradS : 6.4082e-07
#> Iteration: 27 ELBO: -244.8355 Diff params: 0.01308042 Diff ELBO: 0.01234435 
#> gradB : 9.3573e-11 gradD : 1.4703e-03 gradC : 4.5515e-01 gradM : 3.3130e-04 gradS : 6.2759e-05
#> gradB : 2.3108e-03 gradD : 2.1498e-11 gradC : 4.5459e-01 gradM : 3.3231e-04 gradS : 6.2826e-05
#> gradB : 6.0573e-01 gradD : 1.4360e-06 gradC : 3.2510e-10 gradM : 3.3641e-03 gradS : 3.1403e-03
#> gradB : 2.3231e-01 gradD : 8.8221e-05 gradC : 2.0873e-01 gradM : 7.7506e-08 gradS : 9.1084e-04
#> gradB : 1.8566e-01 gradD : 2.1907e-04 gradC : 2.5605e-01 gradM : 3.3341e-06 gradS : 1.1848e-04
#> Iteration: 28 ELBO: -244.8248 Diff params: 0.01206129 Diff ELBO: 0.01071155 
#> gradB : 7.2655e-11 gradD : 1.4014e-03 gradC : 3.9175e-01 gradM : 2.7102e-04 gradS : 2.9491e-04
#> gradB : 2.2318e-03 gradD : 1.8951e-11 gradC : 3.9127e-01 gradM : 2.7196e-04 gradS : 2.9429e-04
#> gradB : 5.2099e-01 gradD : 1.4760e-06 gradC : 5.3583e-09 gradM : 2.9669e-03 gradS : 1.8926e-03
#> gradB : 1.9050e-01 gradD : 7.6809e-05 gradC : 1.8164e-01 gradM : 1.3741e-07 gradS : 4.2306e-04
#> gradB : 1.5859e-01 gradD : 1.3830e-04 gradC : 2.1368e-01 gradM : 2.3712e-06 gradS : 6.5786e-06
#> Iteration: 29 ELBO: -244.8158 Diff params: 0.01121817 Diff ELBO: 0.009002639 
#> gradB : 6.2837e-11 gradD : 1.0710e-03 gradC : 3.2823e-01 gradM : 2.2171e-04 gradS : 5.4163e-05
#> gradB : 1.7055e-03 gradD : 4.5003e-13 gradC : 3.2787e-01 gradM : 2.2256e-04 gradS : 5.3982e-05
#> gradB : 4.3251e-01 gradD : 1.1541e-06 gradC : 5.1732e-09 gradM : 2.5672e-03 gradS : 2.3723e-03
#> gradB : 1.5864e-01 gradD : 7.4516e-05 gradC : 1.5439e-01 gradM : 6.7031e-08 gradS : 6.5320e-04
#> gradB : 1.2104e-01 gradD : 2.7054e-04 gradC : 1.9126e-01 gradM : 4.4102e-06 gradS : 4.8089e-04
#> Iteration: 30 ELBO: -244.8077 Diff params: 0.01034276 Diff ELBO: 0.008092348 
#> gradB : 4.5321e-11 gradD : 1.1960e-03 gradC : 2.8568e-01 gradM : 1.8535e-04 gradS : 5.8661e-04
#> gradB : 1.9031e-03 gradD : 4.6683e-12 gradC : 2.8532e-01 gradM : 1.8619e-04 gradS : 5.8513e-04
#> gradB : 3.9942e-01 gradD : 1.3343e-06 gradC : 3.1873e-09 gradM : 2.3086e-03 gradS : 2.3834e-03
#> gradB : 1.3750e-01 gradD : 5.1022e-05 gradC : 1.3694e-01 gradM : 1.6719e-07 gradS : 8.8214e-04
#> gradB : 1.2530e-01 gradD : 5.1336e-05 gradC : 1.5398e-01 gradM : 2.4062e-06 gradS : 4.1306e-06
#> Iteration: 31 ELBO: -244.8012 Diff params: 0.009753564 Diff ELBO: 0.006511363 
#> gradB : 4.8544e-11 gradD : 6.6180e-04 gradC : 2.3933e-01 gradM : 1.7161e-04 gradS : 2.4868e-05
#> gradB : 1.0670e-03 gradD : 3.6248e-12 gradC : 2.3911e-01 gradM : 1.7232e-04 gradS : 2.4638e-05
#> gradB : 3.0476e-01 gradD : 5.7980e-07 gradC : 1.9772e-10 gradM : 1.8910e-03 gradS : 1.9303e-03
#> gradB : 1.1090e-01 gradD : 6.5212e-05 gradC : 1.1211e-01 gradM : 3.9392e-08 gradS : 5.7721e-04
#> gradB : 8.8092e-02 gradD : 1.0408e-04 gradC : 1.3644e-01 gradM : 1.6089e-06 gradS : 2.8556e-06
#> Iteration: 32 ELBO: -244.7955 Diff params: 0.008736074 Diff ELBO: 0.005690633 
#> gradB : 3.2622e-11 gradD : 6.5886e-04 gradC : 2.0268e-01 gradM : 1.2823e-04 gradS : 2.1697e-05
#> gradB : 1.0684e-03 gradD : 4.7077e-12 gradC : 2.0249e-01 gradM : 1.2869e-04 gradS : 2.1550e-05
#> gradB : 2.6082e-01 gradD : 5.9487e-07 gradC : 2.5632e-11 gradM : 1.6941e-03 gradS : 1.6462e-03
#> gradB : 9.0966e-02 gradD : 4.2191e-05 gradC : 9.8137e-02 gradM : 5.4342e-08 gradS : 4.7150e-04
#> gradB : 7.4400e-02 gradD : 8.1513e-05 gradC : 1.1673e-01 gradM : 1.1596e-06 gradS : 5.5269e-08
#> Iteration: 33 ELBO: -244.7906 Diff params: 0.008208854 Diff ELBO: 0.004921816 
#> gradB : 2.7796e-11 gradD : 5.4118e-04 gradC : 1.7323e-01 gradM : 1.0958e-04 gradS : 2.1798e-05
#> gradB : 8.8077e-04 gradD : 3.7233e-12 gradC : 1.7307e-01 gradM : 1.0997e-04 gradS : 2.1728e-05
#> gradB : 2.2490e-01 gradD : 4.9776e-07 gradC : 3.1263e-12 gradM : 1.4721e-03 gradS : 1.4050e-03
#> gradB : 7.5708e-02 gradD : 3.1901e-05 gradC : 8.3821e-02 gradM : 1.7922e-07 gradS : 3.9508e-04
#> gradB : 6.0570e-02 gradD : 6.8640e-05 gradC : 1.0049e-01 gradM : 1.3039e-06 gradS : 3.7709e-05
#> Iteration: 34 ELBO: -244.7863 Diff params: 0.007564407 Diff ELBO: 0.004276023 
#> gradB : 2.3000e-11 gradD : 4.4821e-04 gradC : 1.4792e-01 gradM : 9.0201e-05 gradS : 8.2980e-05
#> gradB : 7.3440e-04 gradD : 1.4606e-10 gradC : 1.4779e-01 gradM : 9.0584e-05 gradS : 8.2900e-05
#> gradB : 1.9382e-01 gradD : 4.3984e-07 gradC : 1.2523e-09 gradM : 1.2904e-03 gradS : 1.0877e-03
#> gradB : 6.3958e-02 gradD : 2.9959e-05 gradC : 7.2686e-02 gradM : 1.8099e-07 gradS : 2.8864e-04
#> gradB : 5.3573e-02 gradD : 7.3799e-05 gradC : 8.7064e-02 gradM : 1.1101e-06 gradS : 9.6770e-05
#> Iteration: 35 ELBO: -244.7825 Diff params: 0.007038074 Diff ELBO: 0.003781176 
#> gradB : 4.3813e-10 gradD : 4.2584e-04 gradC : 1.2760e-01 gradM : 7.2451e-05 gradS : 8.9379e-05
#> gradB : 6.9754e-04 gradD : 1.1272e-10 gradC : 1.2750e-01 gradM : 7.2815e-05 gradS : 8.9169e-05
#> gradB : 1.5996e-01 gradD : 3.9566e-07 gradC : 4.3540e-10 gradM : 1.1418e-03 gradS : 1.2067e-03
#> gradB : 5.3171e-02 gradD : 3.0417e-05 gradC : 6.4055e-02 gradM : 6.2050e-08 gradS : 4.2363e-04
#> gradB : 4.2545e-02 gradD : 4.8138e-05 gradC : 7.6323e-02 gradM : 1.1380e-06 gradS : 1.4748e-05
#> Iteration: 36 ELBO: -244.7795 Diff params: 0.006612931 Diff ELBO: 0.003075114 
#> gradB : 3.6516e-10 gradD : 3.1530e-04 gradC : 1.1070e-01 gradM : 6.6626e-05 gradS : 2.6783e-05
#> gradB : 5.2020e-04 gradD : 7.4284e-11 gradC : 1.1061e-01 gradM : 6.6888e-05 gradS : 2.6713e-05
#> gradB : 1.4250e-01 gradD : 2.9455e-07 gradC : 6.4222e-11 gradM : 9.9286e-04 gradS : 9.2120e-04
#> gradB : 4.5543e-02 gradD : 2.3020e-05 gradC : 5.4788e-02 gradM : 1.4088e-07 gradS : 2.4344e-04
#> gradB : 3.6306e-02 gradD : 4.7785e-05 gradC : 6.4924e-02 gradM : 5.9641e-07 gradS : 9.9952e-07
#> Iteration: 37 ELBO: -244.7767 Diff params: 0.006105059 Diff ELBO: 0.002778937 
#> gradB : 2.7763e-10 gradD : 2.8039e-04 gradC : 9.3719e-02 gradM : 5.1767e-05 gradS : 6.9409e-06
#> gradB : 4.6406e-04 gradD : 3.8586e-11 gradC : 9.3643e-02 gradM : 5.1975e-05 gradS : 6.8404e-06
#> gradB : 1.2162e-01 gradD : 2.8031e-07 gradC : 1.7274e-10 gradM : 8.6896e-04 gradS : 8.6717e-04
#> gradB : 3.8267e-02 gradD : 1.7198e-05 gradC : 4.7300e-02 gradM : 2.1089e-07 gradS : 2.4679e-04
#> gradB : 2.8809e-02 gradD : 5.7174e-05 gradC : 5.7869e-02 gradM : 1.1512e-06 gradS : 1.1045e-04
#> Iteration: 38 ELBO: -244.7741 Diff params: 0.005662643 Diff ELBO: 0.002574767 
#> gradB : 2.3952e-10 gradD : 2.6930e-04 gradC : 8.2517e-02 gradM : 4.4650e-05 gradS : 1.4430e-04
#> gradB : 4.4480e-04 gradD : 2.5467e-14 gradC : 8.2446e-02 gradM : 4.4891e-05 gradS : 1.4398e-04
#> gradB : 1.1157e-01 gradD : 3.0589e-07 gradC : 1.1786e-10 gradM : 7.7734e-04 gradS : 6.9740e-04
#> gradB : 3.3573e-02 gradD : 1.4452e-05 gradC : 4.2823e-02 gradM : 5.2465e-08 gradS : 2.2340e-04
#> gradB : 3.0292e-02 gradD : 1.5138e-05 gradC : 4.8261e-02 gradM : 6.4746e-07 gradS : 6.4647e-07
#> Iteration: 39 ELBO: -244.7722 Diff params: 0.005409685 Diff ELBO: 0.001936114 
#> gradB : 2.4786e-10 gradD : 1.6751e-04 gradC : 7.0909e-02 gradM : 4.3622e-05 gradS : 8.8484e-06
#> gradB : 2.7971e-04 gradD : 4.1987e-14 gradC : 7.0863e-02 gradM : 4.3823e-05 gradS : 8.8323e-06
#> gradB : 8.7994e-02 gradD : 1.4250e-07 gradC : 1.2492e-11 gradM : 6.4408e-04 gradS : 6.3371e-04
#> gradB : 2.7930e-02 gradD : 1.7982e-05 gradC : 3.4718e-02 gradM : 1.5355e-07 gradS : 1.8110e-04
#> gradB : 2.0170e-02 gradD : 4.0544e-05 gradC : 4.4693e-02 gradM : 8.2394e-07 gradS : 2.0979e-04
#> Iteration: 40 ELBO: -244.7703 Diff params: 0.004824509 Diff ELBO: 0.001877735 
#> gradB : 1.6248e-10 gradD : 1.8885e-04 gradC : 6.2356e-02 gradM : 3.2404e-05 gradS : 2.7807e-04
#> gradB : 3.1601e-04 gradD : 7.0374e-11 gradC : 6.2310e-02 gradM : 3.2560e-05 gradS : 2.7788e-04
#> gradB : 7.9841e-02 gradD : 1.8087e-07 gradC : 4.3928e-11 gradM : 5.9834e-04 gradS : 1.4273e-04
#> gradB : 2.3745e-02 gradD : 1.2700e-05 gradC : 3.1875e-02 gradM : 1.5071e-07 gradS : 6.1176e-05
#> gradB : 2.0255e-02 gradD : 1.9631e-05 gradC : 3.5836e-02 gradM : 6.4676e-07 gradS : 6.8076e-07
#> Iteration: 41 ELBO: -244.7687 Diff params: 0.004640924 Diff ELBO: 0.001567344 
#> gradB : 1.6839e-10 gradD : 1.4037e-04 gradC : 5.1868e-02 gradM : 2.6177e-05 gradS : 4.8449e-06
#> gradB : 2.3457e-04 gradD : 4.0720e-10 gradC : 5.1832e-02 gradM : 2.6305e-05 gradS : 4.8179e-06
#> gradB : 6.6517e-02 gradD : 1.2360e-07 gradC : 1.1823e-09 gradM : 5.1185e-04 gradS : 4.9567e-04
#> gradB : 2.0033e-02 gradD : 1.1787e-05 gradC : 2.8021e-02 gradM : 2.7304e-08 gradS : 1.3190e-04
#> gradB : 1.5795e-02 gradD : 2.5746e-05 gradC : 3.3076e-02 gradM : 5.8561e-07 gradS : 4.6195e-05
#> Iteration: 42 ELBO: -244.7673 Diff params: 0.004366241 Diff ELBO: 0.001430686 
#> gradB : 1.1271e-10 gradD : 1.3227e-04 gradC : 4.6406e-02 gradM : 2.6372e-05 gradS : 6.6017e-05
#> gradB : 2.2057e-04 gradD : 5.2404e-10 gradC : 4.6373e-02 gradM : 2.6485e-05 gradS : 6.5957e-05
#> gradB : 6.0144e-02 gradD : 1.1062e-07 gradC : 4.4865e-10 gradM : 4.4019e-04 gradS : 3.6354e-04
#> gradB : 1.7514e-02 gradD : 8.2856e-06 gradC : 2.3028e-02 gradM : 2.3409e-07 gradS : 1.1093e-04
#> gradB : 1.3962e-02 gradD : 1.8547e-05 gradC : 2.9452e-02 gradM : 5.8778e-07 gradS : 5.3824e-04
#> Iteration: 43 ELBO: -244.7661 Diff params: 0.0039219 Diff ELBO: 0.001235746 
#> gradB : 1.1674e-10 gradD : 1.0989e-04 gradC : 4.1307e-02 gradM : 2.2622e-05 gradS : 6.2778e-04
#> gradB : 1.8743e-04 gradD : 3.6188e-10 gradC : 4.1282e-02 gradM : 2.2729e-05 gradS : 6.2767e-04
#> gradB : 5.0992e-02 gradD : 1.0381e-07 gradC : 9.2018e-11 gradM : 3.9800e-04 gradS : 6.8493e-05
#> gradB : 1.4625e-02 gradD : 9.5405e-06 gradC : 2.1676e-02 gradM : 3.1876e-08 gradS : 2.4478e-04
#> gradB : 1.2425e-02 gradD : 1.5624e-05 gradC : 2.3269e-02 gradM : 2.9232e-07 gradS : 7.1807e-06
#> Iteration: 44 ELBO: -244.765 Diff params: 0.003825488 Diff ELBO: 0.001068227 
#> gradB : 9.8253e-11 gradD : 9.4398e-05 gradC : 3.3254e-02 gradM : 1.6424e-05 gradS : 4.4450e-06
#> gradB : 1.5831e-04 gradD : 3.3021e-13 gradC : 3.3232e-02 gradM : 1.6494e-05 gradS : 4.4292e-06
#> gradB : 4.2538e-02 gradD : 8.3692e-08 gradC : 3.3029e-09 gradM : 3.3599e-04 gradS : 3.8255e-04
#> gradB : 1.2723e-02 gradD : 6.3130e-06 gradC : 1.7113e-02 gradM : 2.6652e-07 gradS : 1.2381e-04
#> gradB : 8.8732e-03 gradD : 1.4993e-05 gradC : 2.2060e-02 gradM : 7.0884e-07 gradS : 2.6358e-05
#> Iteration: 45 ELBO: -244.764 Diff params: 0.003386371 Diff ELBO: 0.0009548651 
#> gradB : 7.4374e-11 gradD : 7.7261e-05 gradC : 3.0307e-02 gradM : 1.3485e-05 gradS : 3.8911e-05
#> gradB : 1.3043e-04 gradD : 2.7867e-13 gradC : 3.0288e-02 gradM : 1.3561e-05 gradS : 3.8846e-05
#> gradB : 3.8828e-02 gradD : 9.2205e-08 gradC : 5.7287e-09 gradM : 3.1704e-04 gradS : 2.3987e-04
#> gradB : 1.0921e-02 gradD : 5.9475e-06 gradC : 1.5948e-02 gradM : 2.7557e-07 gradS : 6.3691e-05
#> gradB : 9.5976e-03 gradD : 1.5495e-05 gradC : 1.8369e-02 gradM : 1.4614e-06 gradS : 1.0459e-04
#> Iteration: 46 ELBO: -244.7632 Diff params: 0.003281322 Diff ELBO: 0.0008475904 
#> gradB : 7.5596e-11 gradD : 8.1079e-05 gradC : 2.6093e-02 gradM : 1.3208e-05 gradS : 1.1067e-04
#> gradB : 1.3777e-04 gradD : 1.2484e-11 gradC : 2.6077e-02 gradM : 1.3274e-05 gradS : 1.1059e-04
#> gradB : 3.3111e-02 gradD : 8.2141e-08 gradC : 1.0889e-09 gradM : 2.7115e-04 gradS : 3.4911e-04
#> gradB : 9.1754e-03 gradD : 5.9699e-06 gradC : 1.4417e-02 gradM : 2.8166e-08 gradS : 1.6280e-04
#> gradB : 7.3023e-03 gradD : 6.9441e-06 gradC : 1.6587e-02 gradM : 6.8889e-07 gradS : 5.8785e-08
#> Iteration: 47 ELBO: -244.7625 Diff params: 0.003128647 Diff ELBO: 0.0006857808 
#> gradB : 5.9534e-11 gradD : 4.9989e-05 gradC : 2.2987e-02 gradM : 1.2433e-05 gradS : 2.3525e-06
#> gradB : 8.4367e-05 gradD : 4.5293e-12 gradC : 2.2975e-02 gradM : 1.2480e-05 gradS : 2.3429e-06
#> gradB : 2.8291e-02 gradD : 4.4204e-08 gradC : 1.2611e-09 gradM : 2.2867e-04 gradS : 2.2533e-04
#> gradB : 7.9622e-03 gradD : 5.1487e-06 gradC : 1.1950e-02 gradM : 3.0512e-08 gradS : 6.2001e-05
#> gradB : 6.3701e-03 gradD : 1.0110e-05 gradC : 1.4075e-02 gradM : 1.3105e-07 gradS : 3.9475e-06
#> Iteration: 48 ELBO: -244.7618 Diff params: 0.002814875 Diff ELBO: 0.0006566915 
#> gradB : 5.0148e-11 gradD : 5.3363e-05 gradC : 1.9568e-02 gradM : 9.1432e-06 gradS : 1.0597e-06
#> gradB : 9.0579e-05 gradD : 4.1464e-12 gradC : 1.9557e-02 gradM : 9.1773e-06 gradS : 1.0504e-06
#> gradB : 2.4725e-02 gradD : 4.8215e-08 gradC : 1.9826e-09 gradM : 2.0217e-04 gradS : 2.4804e-04
#> gradB : 6.7773e-03 gradD : 3.6073e-06 gradC : 1.0763e-02 gradM : 4.5674e-08 gradS : 7.7072e-05
#> gradB : 5.4647e-03 gradD : 7.6087e-06 gradC : 1.2857e-02 gradM : 2.6478e-07 gradS : 2.3866e-06
#> Iteration: 49 ELBO: -244.7613 Diff params: 0.002684324 Diff ELBO: 0.0005761404 
#> gradB : 4.0227e-11 gradD : 4.2873e-05 gradC : 1.7631e-02 gradM : 9.0467e-06 gradS : 3.8410e-06
#> gradB : 7.2823e-05 gradD : 2.2409e-12 gradC : 1.7622e-02 gradM : 9.0754e-06 gradS : 3.8306e-06
#> gradB : 2.1529e-02 gradD : 4.3813e-08 gradC : 6.3152e-10 gradM : 1.7802e-04 gradS : 1.7836e-04
#> gradB : 5.6485e-03 gradD : 1.1788e-05 gradC : 1.0095e-02 gradM : 5.9713e-07 gradS : 4.7208e-05
#> gradB : 4.2348e-03 gradD : 1.9377e-05 gradC : 1.2289e-02 gradM : 1.1304e-06 gradS : 2.3239e-05
#> Iteration: 50 ELBO: -244.7608 Diff params: 0.002559052 Diff ELBO: 0.0005128104 
#> gradB : 3.7412e-11 gradD : 6.1548e-05 gradC : 1.6497e-02 gradM : 9.3957e-06 gradS : 2.5739e-05
#> gradB : 1.0396e-04 gradD : 8.0931e-13 gradC : 1.6487e-02 gradM : 9.3621e-06 gradS : 2.5711e-05
#> gradB : 2.0393e-02 gradD : 3.8167e-08 gradC : 3.5716e-09 gradM : 1.6327e-04 gradS : 1.5958e-04
#> gradB : 5.5446e-03 gradD : 1.6949e-07 gradC : 8.2190e-03 gradM : 8.4025e-09 gradS : 4.3546e-05
#> gradB : 4.2378e-03 gradD : 1.0654e-06 gradC : 9.5491e-03 gradM : 2.8251e-07 gradS : 5.1378e-05
#> Iteration: 51 ELBO: -244.7603 Diff params: 0.002398158 Diff ELBO: 0.0004639873 
#> gradB : 2.4006e-11 gradD : 1.8490e-05 gradC : 1.3222e-02 gradM : 7.0725e-06 gradS : 6.0653e-05
#> gradB : 3.1949e-05 gradD : 1.3576e-12 gradC : 1.3216e-02 gradM : 7.0876e-06 gradS : 6.0643e-05
#> gradB : 1.7555e-02 gradD : 3.7363e-08 gradC : 7.5170e-10 gradM : 1.3465e-04 gradS : 1.2256e-04
#> gradB : 4.5139e-03 gradD : 1.8722e-06 gradC : 7.0401e-03 gradM : 1.3801e-08 gradS : 5.3774e-05
#> gradB : 3.6037e-03 gradD : 3.8271e-06 gradC : 8.5906e-03 gradM : 3.6590e-07 gradS : 1.4438e-06
#> Iteration: 52 ELBO: -244.7599 Diff params: 0.002162273 Diff ELBO: 0.0003824433 
#> gradB : 2.8776e-11 gradD : 2.5684e-05 gradC : 1.1763e-02 gradM : 5.5095e-06 gradS : 3.7747e-06
#> gradB : 4.4374e-05 gradD : 1.1925e-12 gradC : 1.1758e-02 gradM : 5.5299e-06 gradS : 3.7648e-06
#> gradB : 1.4200e-02 gradD : 3.7923e-08 gradC : 3.1672e-09 gradM : 1.2304e-04 gradS : 1.0595e-04
#> gradB : 3.7109e-03 gradD : 2.1760e-06 gradC : 6.4170e-03 gradM : 7.0010e-09 gradS : 2.5224e-05
#> gradB : 3.1508e-03 gradD : 4.8514e-06 gradC : 7.2628e-03 gradM : 9.9562e-08 gradS : 4.5350e-06
#> Iteration: 53 ELBO: -244.7595 Diff params: 0.002080405 Diff ELBO: 0.0003723909 
#> gradB : 2.5396e-11 gradD : 2.6113e-05 gradC : 1.0044e-02 gradM : 5.2825e-06 gradS : 5.8412e-06
#> gradB : 4.4316e-05 gradD : 2.5754e-13 gradC : 1.0039e-02 gradM : 5.3035e-06 gradS : 5.8326e-06
#> gradB : 1.2728e-02 gradD : 2.2085e-08 gradC : 2.5185e-09 gradM : 1.0207e-04 gradS : 1.0097e-04
#> gradB : 3.3494e-03 gradD : 1.7688e-06 gradC : 5.3529e-03 gradM : 1.1454e-08 gradS : 2.8246e-05
#> gradB : 2.5829e-03 gradD : 3.2214e-06 gradC : 6.4179e-03 gradM : 1.5098e-07 gradS : 5.5132e-12
#> Iteration: 54 ELBO: -244.7592 Diff params: 0.001881231 Diff ELBO: 0.0002914798 
#> gradB : 2.0181e-11 gradD : 1.9538e-05 gradC : 8.7551e-03 gradM : 4.2518e-06 gradS : 8.5136e-07
#> gradB : 3.3757e-05 gradD : 3.5893e-13 gradC : 8.7509e-03 gradM : 4.2698e-06 gradS : 8.4809e-07
#> gradB : 1.0837e-02 gradD : 2.2542e-08 gradC : 1.8397e-09 gradM : 9.1708e-05 gradS : 8.9502e-05
#> gradB : 3.6272e-03 gradD : 3.4333e-06 gradC : 3.7902e-03 gradM : 1.2740e-06 gradS : 2.8520e-05
#> gradB : 2.9469e-03 gradD : 6.1216e-06 gradC : 4.5337e-03 gradM : 1.1988e-06 gradS : 5.0816e-06
#> Iteration: 55 ELBO: -244.759 Diff params: 0.001568319 Diff ELBO: 0.0002875969 
#> gradB : 2.6105e-11 gradD : 2.8778e-05 gradC : 6.7825e-03 gradM : 2.2872e-06 gradS : 3.9305e-06
#> gradB : 4.9756e-05 gradD : 3.1457e-13 gradC : 6.7773e-03 gradM : 2.2881e-06 gradS : 3.9250e-06
#> gradB : 9.6680e-03 gradD : 1.5696e-08 gradC : 1.2113e-10 gradM : 8.3025e-05 gradS : 8.8962e-05
#> gradB : 2.1082e-03 gradD : 1.3133e-08 gradC : 4.1600e-03 gradM : 8.6996e-08 gradS : 2.4770e-05
#> gradB : 1.6438e-03 gradD : 3.1193e-07 gradC : 4.9212e-03 gradM : 1.3448e-07 gradS : 3.2544e-10
#> Iteration: 56 ELBO: -244.7587 Diff params: 0.001706219 Diff ELBO: 0.0002347777 
#> gradB : 9.4383e-12 gradD : 6.7214e-06 gradC : 6.4977e-03 gradM : 2.8536e-06 gradS : 5.2291e-07
#> gradB : 1.1552e-05 gradD : 6.3655e-14 gradC : 6.4956e-03 gradM : 2.8749e-06 gradS : 5.2096e-07
#> gradB : 7.6491e-03 gradD : 2.0275e-08 gradC : 9.6289e-10 gradM : 7.0639e-05 gradS : 6.8091e-05
#> gradB : 2.1136e-03 gradD : 2.1382e-06 gradC : 3.4312e-03 gradM : 9.7643e-08 gradS : 1.8667e-05
#> gradB : 1.6991e-03 gradD : 3.8598e-06 gradC : 3.9869e-03 gradM : 1.0274e-07 gradS : 4.1359e-06
#> Iteration: 57 ELBO: -244.7585 Diff params: 0.001499839 Diff ELBO: 0.0002123695 
#> gradB : 1.3492e-11 gradD : 1.6769e-05 gradC : 5.4965e-03 gradM : 2.1858e-06 gradS : 5.4193e-06
#> gradB : 2.8540e-05 gradD : 4.9254e-13 gradC : 5.4934e-03 gradM : 2.2035e-06 gradS : 5.4157e-06
#> gradB : 7.1166e-03 gradD : 1.4051e-08 gradC : 1.3233e-09 gradM : 6.1369e-05 gradS : 5.3637e-05
#> gradB : 1.8325e-03 gradD : 1.2954e-06 gradC : 3.1727e-03 gradM : 4.3766e-09 gradS : 1.5155e-05
#> gradB : 1.3810e-03 gradD : 2.3800e-06 gradC : 3.8561e-03 gradM : 1.0170e-07 gradS : 2.8933e-08
#> Iteration: 58 ELBO: -244.7583 Diff params: 0.001449851 Diff ELBO: 0.0001834098 
#> gradB : 1.0280e-11 gradD : 1.1895e-05 gradC : 5.1617e-03 gradM : 2.2716e-06 gradS : 6.4272e-07
#> gradB : 2.0451e-05 gradD : 2.1525e-13 gradC : 5.1594e-03 gradM : 2.2838e-06 gradS : 6.4061e-07
#> gradB : 6.3153e-03 gradD : 1.4407e-08 gradC : 1.0818e-09 gradM : 5.5868e-05 gradS : 5.2320e-05
#> gradB : 2.6317e-03 gradD : 2.7608e-06 gradC : 2.0566e-03 gradM : 1.6724e-06 gradS : 1.7465e-05
#> gradB : 2.0228e-03 gradD : 4.6050e-06 gradC : 2.8065e-03 gradM : 1.0167e-06 gradS : 7.0846e-05
#> Iteration: 59 ELBO: -244.7581 Diff params: 0.001145681 Diff ELBO: 0.0001852679 
#> gradB : 1.6976e-11 gradD : 2.0210e-05 gradC : 4.2395e-03 gradM : 1.2276e-06 gradS : 8.3879e-05
#> gradB : 3.4825e-05 gradD : 5.2653e-13 gradC : 4.2364e-03 gradM : 1.2167e-06 gradS : 8.3864e-05
#> gradB : 6.0058e-03 gradD : 1.5621e-08 gradC : 1.5095e-09 gradM : 5.3960e-05 gradS : 1.0211e-05
#> gradB : 1.2395e-03 gradD : 5.6583e-09 gradC : 2.6140e-03 gradM : 5.7283e-08 gradS : 3.9814e-05
#> gradB : 1.0174e-03 gradD : 1.4765e-07 gradC : 2.7718e-03 gradM : 1.0912e-07 gradS : 1.1309e-06
#> Iteration: 60 ELBO: -244.758 Diff params: 0.001363958 Diff ELBO: 0.000158497 
#> gradB : 6.1301e-12 gradD : 3.9471e-06 gradC : 3.7147e-03 gradM : 1.2259e-06 gradS : 6.9346e-07
#> gradB : 6.7516e-06 gradD : 1.6001e-13 gradC : 3.7135e-03 gradM : 1.2369e-06 gradS : 6.9299e-07
#> gradB : 4.5310e-03 gradD : 1.1417e-08 gradC : 1.1092e-09 gradM : 4.4018e-05 gradS : 5.0175e-05
#> gradB : 1.1336e-03 gradD : 1.1039e-06 gradC : 2.0967e-03 gradM : 8.1789e-08 gradS : 1.5472e-05
#> gradB : 8.7300e-04 gradD : 2.0675e-06 gradC : 2.4962e-03 gradM : 8.5990e-08 gradS : 8.7898e-07
#> Iteration: 61 ELBO: -244.7579 Diff params: 0.001171255 Diff ELBO: 0.000125562 
#> gradB : 7.1201e-12 gradD : 8.8292e-06 gradC : 3.3443e-03 gradM : 1.0484e-06 gradS : 4.9391e-07
#> gradB : 1.5091e-05 gradD : 4.6525e-13 gradC : 3.3426e-03 gradM : 1.0572e-06 gradS : 4.9148e-07
#> gradB : 4.1708e-03 gradD : 8.2096e-09 gradC : 7.9344e-10 gradM : 3.9347e-05 gradS : 4.4629e-05
#> gradB : 1.3104e-03 gradD : 1.8075e-06 gradC : 1.5453e-03 gradM : 7.4851e-07 gradS : 1.5854e-05
#> gradB : 9.8194e-04 gradD : 2.8983e-06 gradC : 2.0395e-03 gradM : 4.7429e-07 gradS : 2.9262e-05
#> Iteration: 62 ELBO: -244.7577 Diff params: 0.0009964364 Diff ELBO: 0.0001262757 
#> gradB : 8.5539e-12 gradD : 1.1261e-05 gradC : 2.8857e-03 gradM : 7.6876e-07 gradS : 3.5375e-05
#> gradB : 1.9311e-05 gradD : 6.4826e-13 gradC : 2.8838e-03 gradM : 7.6904e-07 gradS : 3.5370e-05
#> gradB : 3.9364e-03 gradD : 8.0115e-09 gradC : 1.4412e-09 gradM : 3.7121e-05 gradS : 2.9453e-06
#> gradB : 7.5413e-04 gradD : 5.3860e-09 gradC : 1.8351e-03 gradM : 3.3447e-08 gradS : 1.1814e-05
#> gradB : 7.0854e-04 gradD : 2.3855e-07 gradC : 1.7994e-03 gradM : 7.6690e-08 gradS : 8.0311e-06
#> Iteration: 63 ELBO: -244.7576 Diff params: 0.001130771 Diff ELBO: 0.0001239708 
#> gradB : 3.9787e-12 gradD : 3.3202e-06 gradC : 2.4305e-03 gradM : 8.9076e-07 gradS : 7.0419e-06
#> gradB : 5.7833e-06 gradD : 8.7334e-14 gradC : 2.4296e-03 gradM : 8.9967e-07 gradS : 7.0397e-06
#> gradB : 2.9801e-03 gradD : 6.3315e-09 gradC : 7.6810e-10 gradM : 2.8246e-05 gradS : 4.9227e-05
#> gradB : 7.1232e-04 gradD : 7.0396e-07 gradC : 1.3865e-03 gradM : 4.7096e-08 gradS : 2.1727e-05
#> gradB : 4.7564e-04 gradD : 1.1448e-06 gradC : 1.8351e-03 gradM : 5.6955e-08 gradS : 1.6319e-06
#> Iteration: 64 ELBO: -244.7575 Diff params: 0.0009539928 Diff ELBO: 8.550268e-05 
#> gradB : 3.9600e-12 gradD : 4.8651e-06 gradC : 2.3636e-03 gradM : 7.8896e-07 gradS : 2.3164e-06
#> gradB : 8.3244e-06 gradD : 3.6293e-13 gradC : 2.3628e-03 gradM : 7.9677e-07 gradS : 2.3132e-06
#> gradB : 2.7870e-03 gradD : 5.9556e-09 gradC : 2.2126e-10 gradM : 2.7420e-05 gradS : 1.9371e-05
#> gradB : 7.3352e-04 gradD : 7.3435e-07 gradC : 1.2501e-03 gradM : 9.3130e-08 gradS : 4.4131e-06
#> gradB : 6.2085e-04 gradD : 1.1916e-06 gradC : 1.4201e-03 gradM : 9.0444e-08 gradS : 1.8784e-07
#> Iteration: 65 ELBO: -244.7574 Diff params: 0.0009074759 Diff ELBO: 8.717552e-05 
#> gradB : 4.6576e-12 gradD : 5.6119e-06 gradC : 1.9542e-03 gradM : 5.7918e-07 gradS : 3.0412e-07
#> gradB : 9.6084e-06 gradD : 4.5087e-13 gradC : 1.9533e-03 gradM : 5.8231e-07 gradS : 3.0316e-07
#> gradB : 2.4484e-03 gradD : 4.7574e-09 gradC : 5.4096e-10 gradM : 2.3582e-05 gradS : 2.0908e-05
#> gradB : 9.7154e-04 gradD : 9.0764e-07 gradC : 8.4022e-04 gradM : 8.3780e-07 gradS : 6.9677e-06
#> gradB : 7.8161e-04 gradD : 1.5966e-06 gradC : 9.8161e-04 gradM : 8.3931e-07 gradS : 4.6338e-06
#> Iteration: 66 ELBO: -244.7573 Diff params: 0.0007256796 Diff ELBO: 8.707962e-05 
#> gradB : 6.8057e-12 gradD : 7.5207e-06 gradC : 1.5247e-03 gradM : 4.5609e-07 gradS : 3.2658e-06
#> gradB : 1.3182e-05 gradD : 2.5028e-13 gradC : 1.5234e-03 gradM : 4.5129e-07 gradS : 3.2643e-06
#> gradB : 2.2919e-03 gradD : 4.5638e-09 gradC : 7.9844e-11 gradM : 2.1817e-05 gradS : 3.2886e-05
#> gradB : 4.2025e-04 gradD : 6.3667e-10 gradC : 1.0903e-03 gradM : 7.3877e-08 gradS : 1.1392e-05
#> gradB : 2.3729e-04 gradD : 8.5861e-08 gradC : 1.4488e-03 gradM : 1.5158e-07 gradS : 4.3824e-06
#> Iteration: 67 ELBO: -244.7573 Diff params: 0.0008918957 Diff ELBO: 8.521505e-05 
#> gradB : 4.7045e-11 gradD : 9.7900e-07 gradC : 1.7399e-03 gradM : 6.4959e-07 gradS : 5.1807e-06
#> gradB : 1.6240e-06 gradD : 1.1866e-12 gradC : 1.7395e-03 gradM : 6.5368e-07 gradS : 5.1785e-06
#> gradB : 2.0043e-03 gradD : 6.0607e-09 gradC : 8.6934e-11 gradM : 1.9956e-05 gradS : 1.2956e-05
#> gradB : 4.5536e-04 gradD : 4.4196e-07 gradC : 8.7470e-04 gradM : 1.2107e-07 gradS : 3.8330e-06
#> gradB : 4.2527e-04 gradD : 7.4663e-07 gradC : 1.0297e-03 gradM : 4.9287e-08 gradS : 3.8276e-05
#> Iteration: 68 ELBO: -244.7572 Diff params: 0.000755497 Diff ELBO: 5.242095e-05 
#> gradB : 3.5977e-12 gradD : 3.7746e-06 gradC : 1.4096e-03 gradM : 4.4073e-07 gradS : 4.2621e-05
#> gradB : 6.5174e-06 gradD : 2.5960e-13 gradC : 1.4089e-03 gradM : 4.4436e-07 gradS : 4.2617e-05
#> gradB : 1.7070e-03 gradD : 2.7266e-09 gradC : 3.7588e-10 gradM : 1.6598e-05 gradS : 8.3010e-06
#> gradB : 3.9593e-04 gradD : 3.5844e-07 gradC : 7.6573e-04 gradM : 5.8922e-08 gradS : 2.1236e-05
#> gradB : 3.2469e-04 gradD : 5.9015e-07 gradC : 8.0849e-04 gradM : 1.3857e-07 gradS : 1.5272e-07
#> Iteration: 69 ELBO: -244.7572 Diff params: 0.0007020915 Diff ELBO: 5.90354e-05 
#> gradB : 2.8206e-12 gradD : 2.9519e-06 gradC : 1.1089e-03 gradM : 2.4919e-07 gradS : 1.2088e-07
#> gradB : 5.1064e-06 gradD : 2.2732e-13 gradC : 1.1083e-03 gradM : 2.5176e-07 gradS : 1.2043e-07
#> gradB : 1.4109e-03 gradD : 2.4656e-09 gradC : 2.8161e-10 gradM : 1.4584e-05 gradS : 1.3713e-05
#> gradB : 6.5711e-04 gradD : 6.7815e-07 gradC : 4.0033e-04 gradM : 1.2303e-06 gradS : 5.4774e-06
#> gradB : 5.1605e-04 gradD : 1.6497e-06 gradC : 5.1187e-04 gradM : 1.1294e-06 gradS : 1.3030e-05
#> Iteration: 70 ELBO: -244.7571 Diff params: 0.0004956857 Diff ELBO: 7.820463e-05 
#> gradB : 1.8267e-13 gradD : 6.1416e-06 gradC : 8.3489e-04 gradM : 4.2451e-07 gradS : 1.3625e-05
#> gradB : 1.0655e-05 gradD : 5.5135e-13 gradC : 8.3400e-04 gradM : 4.1985e-07 gradS : 1.3618e-05
#> gradB : 1.3384e-03 gradD : 4.7856e-09 gradC : 1.3186e-10 gradM : 1.4354e-05 gradS : 2.1246e-05
#> gradB : 2.1785e-04 gradD : 1.0380e-09 gradC : 6.9854e-04 gradM : 4.6562e-08 gradS : 1.5625e-05
#> gradB : 1.9627e-04 gradD : 1.0150e-09 gradC : 7.5762e-04 gradM : 2.7282e-08 gradS : 5.5560e-08
#> Iteration: 71 ELBO: -244.757 Diff params: 0.0007145975 Diff ELBO: 2.884762e-05 
#> gradB : 8.0882e-13 gradD : 4.8266e-07 gradC : 9.6202e-04 gradM : 3.5375e-07 gradS : 1.3675e-07
#> gradB : 7.8777e-07 gradD : 2.0492e-10 gradC : 9.6187e-04 gradM : 3.5560e-07 gradS : 1.3647e-07
#> gradB : 1.0323e-03 gradD : 2.7525e-10 gradC : 5.4208e-12 gradM : 1.0952e-05 gradS : 1.0197e-05
#> gradB : 2.2855e-04 gradD : 2.1755e-07 gradC : 4.9428e-04 gradM : 5.5315e-08 gradS : 2.8751e-06
#> gradB : 8.4241e-05 gradD : 3.7585e-07 gradC : 7.6363e-04 gradM : 9.2220e-08 gradS : 4.6754e-06
#> Iteration: 72 ELBO: -244.757 Diff params: 0.000690094 Diff ELBO: 4.151674e-05 
#> gradB : 7.3254e-13 gradD : 1.2026e-06 gradC : 9.0658e-04 gradM : 1.7599e-07 gradS : 5.1325e-06
#> gradB : 2.1277e-06 gradD : 1.4838e-12 gradC : 9.0630e-04 gradM : 1.7811e-07 gradS : 5.1302e-06
#> gradB : 1.0672e-03 gradD : 4.7677e-09 gradC : 2.4063e-10 gradM : 1.2376e-05 gradS : 7.2691e-06
#> gradB : 2.1579e-04 gradD : 1.9917e-07 gradC : 5.4153e-04 gradM : 5.0573e-08 gradS : 2.7275e-06
#> gradB : 2.3306e-04 gradD : 3.6232e-07 gradC : 5.3914e-04 gradM : 4.5568e-08 gradS : 1.7381e-06
#> Iteration: 73 ELBO: -244.757 Diff params: 0.0006043746 Diff ELBO: 3.926027e-05 
#> gradB : 1.8159e-12 gradD : 1.9256e-06 gradC : 7.4156e-04 gradM : 2.0064e-07 gradS : 2.4754e-06
#> gradB : 3.2744e-06 gradD : 2.4394e-12 gradC : 7.4122e-04 gradM : 2.0236e-07 gradS : 2.4744e-06
#> gradB : 9.1482e-04 gradD : 1.1905e-09 gradC : 1.7541e-10 gradM : 9.1655e-06 gradS : 2.9843e-06
#> gradB : 2.1054e-04 gradD : 1.8765e-07 gradC : 3.7401e-04 gradM : 1.2491e-07 gradS : 6.1164e-07
#> gradB : 1.7811e-04 gradD : 3.1039e-07 gradC : 4.5778e-04 gradM : 5.8264e-08 gradS : 2.9017e-05
#> Iteration: 74 ELBO: -244.7569 Diff params: 0.0004916618 Diff ELBO: 3.812493e-05 
#> gradB : 1.1182e-12 gradD : 1.5587e-06 gradC : 6.2000e-04 gradM : 1.4680e-07 gradS : 3.1383e-05
#> gradB : 2.6729e-06 gradD : 1.9693e-12 gradC : 6.1975e-04 gradM : 1.4827e-07 gradS : 3.1383e-05
#> gradB : 7.4638e-04 gradD : 7.2070e-10 gradC : 1.8949e-10 gradM : 8.0313e-06 gradS : 9.7995e-06
#> gradB : 1.5548e-04 gradD : 1.0759e-07 gradC : 3.9310e-04 gradM : 5.7354e-09 gradS : 1.9569e-05
#> gradB : 1.3101e-04 gradD : 2.8303e-07 gradC : 4.2988e-04 gradM : 4.1457e-08 gradS : 1.9229e-06
#> Iteration: 75 ELBO: -244.7569 Diff params: 0.0005050508 Diff ELBO: 3.68439e-05 
#> gradB : 1.0744e-12 gradD : 1.2737e-06 gradC : 5.6150e-04 gradM : 1.3120e-07 gradS : 2.2795e-06
#> gradB : 2.2558e-06 gradD : 1.5909e-12 gradC : 5.6131e-04 gradM : 1.3187e-07 gradS : 2.2787e-06
#> gradB : 6.1774e-04 gradD : 1.4230e-09 gradC : 3.7812e-13 gradM : 7.2182e-06 gradS : 3.5042e-06
#> gradB : 5.6820e-05 gradD : 5.1121e-07 gradC : 6.0780e-04 gradM : 6.8198e-07 gradS : 1.5746e-06
#> gradB : 3.1109e-05 gradD : 6.5432e-07 gradC : 6.7109e-04 gradM : 6.7643e-07 gradS : 2.2276e-08
#> Iteration: 76 ELBO: -244.7569 Diff params: 0.0006451389 Diff ELBO: 2.332219e-05 
#> gradB : 5.5680e-10 gradD : 9.9854e-07 gradC : 7.2280e-04 gradM : 8.6097e-07 gradS : 3.3048e-08
#> gradB : 1.5561e-06 gradD : 1.0622e-12 gradC : 7.2265e-04 gradM : 8.6062e-07 gradS : 3.2795e-08
#> gradB : 7.4027e-04 gradD : 2.4901e-10 gradC : 6.5836e-11 gradM : 5.7196e-06 gradS : 8.7791e-06
#> gradB : 1.3270e-04 gradD : 3.2784e-09 gradC : 2.8818e-04 gradM : 2.4354e-07 gradS : 3.2905e-06
#> gradB : 6.6795e-05 gradD : 4.6569e-08 gradC : 4.6326e-04 gradM : 2.4319e-07 gradS : 6.8419e-05
#> Iteration: 77 ELBO: -244.7568 Diff params: 0.0004881153 Diff ELBO: 4.169445e-05 
#> gradB : 4.7308e-13 gradD : 4.5182e-07 gradC : 5.6867e-04 gradM : 4.6216e-07 gradS : 7.0976e-05
#> gradB : 8.3647e-07 gradD : 1.3870e-10 gradC : 5.6850e-04 gradM : 4.6289e-07 gradS : 7.0974e-05
#> gradB : 6.8088e-04 gradD : 8.0102e-10 gradC : 1.5085e-10 gradM : 6.3458e-06 gradS : 3.4440e-05
#> gradB : 1.4159e-04 gradD : 5.1190e-08 gradC : 2.9409e-04 gradM : 5.7921e-09 gradS : 5.0492e-05
#> gradB : 1.3304e-04 gradD : 1.0853e-07 gradC : 3.0402e-04 gradM : 3.3411e-08 gradS : 2.0038e-05
#> Iteration: 78 ELBO: -244.7568 Diff params: 0.0004544496 Diff ELBO: 2.63973e-05 
#> gradB : 1.0055e-12 gradD : 8.5098e-07 gradC : 4.1707e-04 gradM : 1.5972e-07 gradS : 2.1757e-05
#> gradB : 1.5041e-06 gradD : 1.0772e-12 gradC : 4.1692e-04 gradM : 1.6061e-07 gradS : 2.1756e-05
#> gradB : 5.0112e-04 gradD : 1.3395e-09 gradC : 1.5100e-10 gradM : 4.9345e-06 gradS : 6.6285e-06
#> gradB : 9.7772e-05 gradD : 7.8518e-07 gradC : 2.1654e-04 gradM : 1.2782e-07 gradS : 1.2642e-05
#> gradB : 9.0750e-05 gradD : 9.9520e-07 gradC : 2.1570e-04 gradM : 1.6385e-07 gradS : 2.7303e-09
#> Iteration: 79 ELBO: -244.7568 Diff params: 0.000563325 Diff ELBO: 2.385808e-05 
#> gradB : 7.6621e-13 gradD : 2.2825e-06 gradC : 3.0067e-04 gradM : 1.6888e-07 gradS : 4.9983e-08
#> gradB : 3.8318e-06 gradD : 2.9023e-12 gradC : 3.0040e-04 gradM : 1.6721e-07 gradS : 4.9541e-08
#> gradB : 4.2430e-04 gradD : 2.8389e-10 gradC : 1.2073e-11 gradM : 4.3385e-06 gradS : 3.1049e-06
#> gradB : 9.4152e-05 gradD : 7.4290e-10 gradC : 1.7582e-04 gradM : 1.3874e-07 gradS : 7.5140e-07
#> gradB : 4.6176e-05 gradD : 2.0724e-08 gradC : 3.1736e-04 gradM : 1.3197e-07 gradS : 1.0753e-04
#> Iteration: 80 ELBO: -244.7567 Diff params: 0.0003576795 Diff ELBO: 2.921945e-05 
#> gradB : 9.5263e-12 gradD : 2.2341e-07 gradC : 3.8059e-04 gradM : 2.6888e-07 gradS : 1.0968e-04
#> gradB : 3.8271e-07 gradD : 9.2223e-11 gradC : 3.8051e-04 gradM : 2.6951e-07 gradS : 1.0968e-04
#> gradB : 4.2510e-04 gradD : 2.5796e-10 gradC : 1.8399e-10 gradM : 4.3175e-06 gradS : 6.9752e-05
#> gradB : 6.2328e-05 gradD : 2.7746e-10 gradC : 2.3521e-04 gradM : 5.3901e-08 gradS : 8.9246e-05
#> gradB : 7.4607e-05 gradD : 5.4227e-09 gradC : 1.6495e-04 gradM : 6.7940e-08 gradS : 6.1158e-07
#> Iteration: 81 ELBO: -244.7567 Diff params: 0.0003927367 Diff ELBO: 1.559882e-05 
#> gradB : 6.2943e-13 gradD : 2.7573e-07 gradC : 2.2986e-04 gradM : 1.2803e-07 gradS : 5.7825e-07
#> gradB : 4.9102e-07 gradD : 1.0114e-10 gradC : 2.2978e-04 gradM : 1.2902e-07 gradS : 5.7808e-07
#> gradB : 2.8585e-04 gradD : 4.3345e-10 gradC : 2.8364e-11 gradM : 2.9977e-06 gradS : 3.7917e-06
#> gradB : 6.2357e-05 gradD : 2.9528e-09 gradC : 1.4996e-04 gradM : 1.2276e-08 gradS : 1.5339e-06
#> gradB : 4.6452e-05 gradD : 1.7850e-08 gradC : 1.8034e-04 gradM : 1.9123e-08 gradS : 2.1843e-09
#> Iteration: 82 ELBO: -244.7567 Diff params: 0.000311598 Diff ELBO: 1.584532e-05 
#> gradB : 1.5673e-11 gradD : 2.4849e-07 gradC : 2.3314e-04 gradM : 1.1520e-07 gradS : 1.9527e-08
#> gradB : 4.3687e-07 gradD : 9.3257e-11 gradC : 2.3308e-04 gradM : 1.1619e-07 gradS : 1.9429e-08
#> gradB : 2.6925e-04 gradD : 3.8267e-10 gradC : 3.7094e-11 gradM : 2.6342e-06 gradS : 2.6143e-06
#> gradB : 7.9461e-05 gradD : 5.2655e-08 gradC : 1.4425e-04 gradM : 9.3080e-09 gradS : 6.5359e-07
#> gradB : 6.2725e-05 gradD : 9.0349e-08 gradC : 1.6212e-04 gradM : 7.5648e-09 gradS : 5.4462e-06
#> Iteration: 83 ELBO: -244.7567 Diff params: 0.0003113739 Diff ELBO: 1.459581e-05 
#> gradB : 3.6024e-14 gradD : 5.0298e-07 gradC : 2.1934e-04 gradM : 1.2634e-07 gradS : 4.9338e-06
#> gradB : 8.7863e-07 gradD : 1.9695e-10 gradC : 2.1925e-04 gradM : 1.2702e-07 gradS : 4.9342e-06
#> gradB : 2.7242e-04 gradD : 3.7366e-10 gradC : 2.1295e-11 gradM : 2.2878e-06 gradS : 1.3630e-05
#> gradB : 7.9495e-05 gradD : 6.4506e-08 gradC : 1.0042e-04 gradM : 1.5964e-08 gradS : 8.9649e-06
#> gradB : 6.1559e-05 gradD : 1.1792e-07 gradC : 1.2869e-04 gradM : 6.0064e-09 gradS : 2.0099e-09
#> Iteration: 84 ELBO: -244.7567 Diff params: 0.000253868 Diff ELBO: 1.547901e-05 
#> gradB : 3.4459e-13 gradD : 5.4187e-07 gradC : 1.7823e-04 gradM : 7.1195e-08 gradS : 3.1088e-08
#> gradB : 9.1976e-07 gradD : 2.2557e-10 gradC : 1.7814e-04 gradM : 7.1621e-08 gradS : 3.0996e-08
#> gradB : 2.2789e-04 gradD : 1.9648e-10 gradC : 4.3437e-11 gradM : 2.0847e-06 gradS : 1.8173e-06
#> gradB : 5.3143e-05 gradD : 3.6880e-08 gradC : 1.1197e-04 gradM : 4.3241e-09 gradS : 3.9588e-07
#> gradB : 4.4417e-05 gradD : 7.4563e-08 gradC : 1.2532e-04 gradM : 7.0875e-09 gradS : 3.6235e-07
#> Iteration: 85 ELBO: -244.7567 Diff params: 0.0002754486 Diff ELBO: 1.395428e-05 
#> gradB : 1.8536e-13 gradD : 3.6160e-07 gradC : 1.6557e-04 gradM : 9.6310e-08 gradS : 3.3979e-07
#> gradB : 6.1327e-07 gradD : 1.5101e-10 gradC : 1.6551e-04 gradM : 9.6773e-08 gradS : 3.3975e-07
#> gradB : 1.9968e-04 gradD : 1.2939e-10 gradC : 1.3145e-10 gradM : 1.7156e-06 gradS : 2.6176e-06
#> gradB : 5.0402e-05 gradD : 4.6822e-08 gradC : 9.4598e-05 gradM : 7.9064e-09 gradS : 1.1146e-06
#> gradB : 3.6912e-05 gradD : 9.2470e-08 gradC : 1.1312e-04 gradM : 1.7882e-08 gradS : 7.9638e-07
#> Iteration: 86 ELBO: -244.7567 Diff params: 0.0002447522 Diff ELBO: 1.422429e-05 
#> gradB : 3.1664e-13 gradD : 3.9979e-07 gradC : 1.5141e-04 gradM : 9.3840e-08 gradS : 6.4885e-07
#> gradB : 7.0146e-07 gradD : 1.5527e-10 gradC : 1.5134e-04 gradM : 9.4046e-08 gradS : 6.4877e-07
#> gradB : 1.9356e-04 gradD : 1.0988e-10 gradC : 8.3235e-11 gradM : 1.5742e-06 gradS : 4.3165e-06
#> gradB : 4.5386e-05 gradD : 1.4335e-09 gradC : 7.7108e-05 gradM : 1.2140e-09 gradS : 2.0985e-06
#> gradB : 4.2304e-05 gradD : 2.1981e-08 gradC : 7.4630e-05 gradM : 7.0611e-08 gradS : 5.5702e-05
#> Iteration: 87 ELBO: -244.7566 Diff params: 0.0003645483 Diff ELBO: 1.596725e-05 
#> gradB : 2.6341e-13 gradD : 2.2581e-07 gradC : 1.0598e-04 gradM : 8.6602e-08 gradS : 5.4351e-05
#> gradB : 3.8258e-07 gradD : 9.4647e-11 gradC : 1.0594e-04 gradM : 8.7123e-08 gradS : 5.4352e-05
#> gradB : 1.2949e-04 gradD : 1.7904e-10 gradC : 1.9482e-11 gradM : 1.4683e-06 gradS : 7.0521e-05
#> gradB : 3.2831e-05 gradD : 3.4779e-08 gradC : 7.6938e-05 gradM : 8.0682e-09 gradS : 6.0197e-05
#> gradB : 2.3391e-05 gradD : 3.5715e-08 gradC : 1.0977e-04 gradM : 1.0545e-07 gradS : 8.0214e-08
#> Iteration: 88 ELBO: -244.7566 Diff params: 0.0002863803 Diff ELBO: 8.652818e-06 
#> gradB : 2.5687e-12 gradD : 1.9116e-07 gradC : 1.3755e-04 gradM : 2.1310e-07 gradS : 8.6310e-08
#> gradB : 3.2632e-07 gradD : 7.8712e-11 gradC : 1.3751e-04 gradM : 2.1360e-07 gradS : 8.6423e-08
#> gradB : 1.5185e-04 gradD : 1.6072e-11 gradC : 4.8949e-14 gradM : 1.1098e-06 gradS : 1.5906e-06
#> gradB : 1.9687e-05 gradD : 6.6917e-08 gradC : 9.7044e-05 gradM : 1.0790e-07 gradS : 3.9776e-07
#> gradB : 9.4568e-06 gradD : 1.0717e-07 gradC : 1.3396e-04 gradM : 1.6477e-07 gradS : 6.6493e-06
#> Iteration: 89 ELBO: -244.7566 Diff params: 0.0002488502 Diff ELBO: 8.359113e-06 
#> gradB : 6.6047e-10 gradD : 2.1093e-07 gradC : 1.4995e-04 gradM : 2.3643e-07 gradS : 6.8640e-06
#> gradB : 3.4824e-07 gradD : 9.0551e-11 gradC : 1.4992e-04 gradM : 2.3672e-07 gradS : 6.8637e-06
#> gradB : 1.5496e-04 gradD : 4.1710e-11 gradC : 1.7742e-11 gradM : 1.0564e-06 gradS : 1.6267e-06
#> gradB : 4.7156e-05 gradD : 5.1620e-08 gradC : 4.9860e-05 gradM : 3.1943e-08 gradS : 3.1842e-06
#> gradB : 4.3680e-05 gradD : 8.3493e-08 gradC : 4.6256e-05 gradM : 4.0494e-08 gradS : 9.9761e-07
#> Iteration: 90 ELBO: -244.7566 Diff params: 0.0001872307 Diff ELBO: 1.127375e-05 
#> gradB : 1.0882e-09 gradD : 3.6790e-07 gradC : 7.1665e-05 gradM : 6.0292e-08 gradS : 8.1868e-07
#> gradB : 6.4454e-07 gradD : 1.5771e-10 gradC : 7.1610e-05 gradM : 6.0323e-08 gradS : 8.1876e-07
#> gradB : 1.0620e-04 gradD : 6.5424e-11 gradC : 6.8157e-10 gradM : 8.9579e-07 gradS : 3.3074e-06
#> gradB : 5.6777e-05 gradD : 6.0880e-09 gradC : 2.2783e-05 gradM : 8.5504e-08 gradS : 2.2384e-06
#> gradB : 4.6277e-05 gradD : 1.8975e-08 gradC : 3.1499e-05 gradM : 6.5744e-08 gradS : 1.3257e-09
#> Iteration: 91 ELBO: -244.7566 Diff params: 0.0001146337 Diff ELBO: 1.016672e-05 
#> gradB : 6.4336e-12 gradD : 2.6152e-07 gradC : 5.6734e-05 gradM : 2.1982e-08 gradS : 1.9712e-08
#> gradB : 4.6186e-07 gradD : 9.9382e-11 gradC : 5.6689e-05 gradM : 2.1995e-08 gradS : 1.9660e-08
#> gradB : 8.9893e-05 gradD : 9.8860e-11 gradC : 2.0908e-11 gradM : 8.7076e-07 gradS : 5.3359e-07
#> gradB : 1.9892e-05 gradD : 2.8766e-11 gradC : 3.7613e-05 gradM : 9.0299e-09 gradS : 1.0867e-07
#> gradB : 1.5045e-05 gradD : 1.2159e-08 gradC : 5.9842e-05 gradM : 4.1688e-08 gradS : 4.5420e-05
#> Iteration: 92 ELBO: -244.7566 Diff params: 0.0001806926 Diff ELBO: 1.191078e-05 
#> gradB : 4.8916e-11 gradD : 7.1949e-08 gradC : 7.3022e-05 gradM : 7.3478e-08 gradS : 4.5971e-05
#> gradB : 1.1866e-07 gradD : 3.2041e-11 gradC : 7.3007e-05 gradM : 7.3781e-08 gradS : 4.5970e-05
#> gradB : 8.1248e-05 gradD : 8.0001e-11 gradC : 1.6943e-10 gradM : 7.2972e-07 gradS : 3.5051e-05
#> gradB : 2.5325e-05 gradD : 2.3466e-08 gradC : 3.3500e-05 gradM : 4.2984e-09 gradS : 3.9374e-05
#> gradB : 2.7551e-05 gradD : 2.3363e-08 gradC : 2.4422e-05 gradM : 6.0739e-08 gradS : 2.5207e-08
#> Iteration: 93 ELBO: -244.7566 Diff params: 0.0001836659 Diff ELBO: 4.72862e-06 
#> gradB : 2.7816e-12 gradD : 1.9773e-07 gradC : 4.1453e-05 gradM : 3.3731e-08 gradS : 1.0198e-08
#> gradB : 3.5058e-07 gradD : 7.4381e-11 gradC : 4.1419e-05 gradM : 3.3949e-08 gradS : 1.0167e-08
#> gradB : 6.2917e-05 gradD : 3.2346e-11 gradC : 2.5602e-10 gradM : 6.6445e-07 gradS : 5.6897e-07
#> gradB : 1.8046e-05 gradD : 1.3496e-08 gradC : 3.2740e-05 gradM : 2.3082e-09 gradS : 1.5753e-07
#> gradB : 1.4831e-05 gradD : 2.3630e-08 gradC : 3.2188e-05 gradM : 3.2027e-09 gradS : 1.1578e-05
#> Iteration: 94 ELBO: -244.7566 Diff params: 0.0001537968 Diff ELBO: 4.528122e-06 
#> gradB : 4.3167e-12 gradD : 1.3093e-07 gradC : 4.5384e-05 gradM : 1.6875e-08 gradS : 1.1140e-05
#> gradB : 2.2761e-07 gradD : 5.1994e-11 gradC : 4.5361e-05 gradM : 1.7006e-08 gradS : 1.1140e-05
#> gradB : 6.0740e-05 gradD : 5.6839e-11 gradC : 2.1695e-11 gradM : 5.3316e-07 gradS : 1.6199e-05
#> gradB : 1.5145e-05 gradD : 5.4810e-09 gradC : 2.5322e-05 gradM : 4.9430e-10 gradS : 1.3447e-05
#> gradB : 1.1110e-05 gradD : 1.4829e-08 gradC : 3.5491e-05 gradM : 4.0481e-09 gradS : 6.6288e-07
#> Iteration: 95 ELBO: -244.7566 Diff params: 0.0001301876 Diff ELBO: 6.25444e-06 
#> gradB : 1.2099e-10 gradD : 7.5004e-08 gradC : 4.5376e-05 gradM : 2.4426e-08 gradS : 5.9051e-07
#> gradB : 1.2951e-07 gradD : 3.2381e-11 gradC : 4.5363e-05 gradM : 2.4585e-08 gradS : 5.9052e-07
#> gradB : 5.3308e-05 gradD : 5.8840e-11 gradC : 1.0062e-11 gradM : 4.7529e-07 gradS : 2.2687e-06
#> gradB : 1.2506e-05 gradD : 1.2236e-08 gradC : 2.4262e-05 gradM : 2.1186e-10 gradS : 1.3333e-06
#> gradB : 7.1056e-06 gradD : 2.5229e-08 gradC : 3.9926e-05 gradM : 8.5142e-09 gradS : 5.5505e-06
#> Iteration: 96 ELBO: -244.7566 Diff params: 0.0001257575 Diff ELBO: 7.183001e-06 
#> gradB : 5.0600e-13 gradD : 8.4959e-08 gradC : 4.8906e-05 gradM : 3.2453e-08 gradS : 5.7443e-06
#> gradB : 1.4098e-07 gradD : 3.6957e-11 gradC : 4.8892e-05 gradM : 3.2607e-08 gradS : 5.7441e-06
#> gradB : 5.5406e-05 gradD : 1.4394e-11 gradC : 2.9135e-10 gradM : 4.7622e-07 gradS : 2.7859e-06
#> gradB : 1.2040e-05 gradD : 1.0386e-08 gradC : 2.4540e-05 gradM : 1.1717e-10 gradS : 4.0354e-06
#> gradB : 1.1927e-05 gradD : 1.6083e-08 gradC : 2.3743e-05 gradM : 5.0141e-09 gradS : 6.9194e-07
#> Iteration: 97 ELBO: -244.7566 Diff params: 0.0001259254 Diff ELBO: 4.064396e-06 
#> gradB : 3.3673e-12 gradD : 1.0104e-07 gradC : 3.4129e-05 gradM : 1.3327e-08 gradS : 6.7543e-07
#> gradB : 1.7948e-07 gradD : 3.7783e-11 gradC : 3.4110e-05 gradM : 1.3449e-08 gradS : 6.7541e-07
#> gradB : 4.5608e-05 gradD : 8.1820e-12 gradC : 6.0854e-14 gradM : 4.0560e-07 gradS : 9.6855e-07
#> gradB : 1.0020e-05 gradD : 4.5803e-09 gradC : 2.0912e-05 gradM : 3.0056e-10 gradS : 6.1824e-07
#> gradB : 9.5390e-06 gradD : 1.4501e-08 gradC : 1.6935e-05 gradM : 9.5792e-09 gradS : 1.8602e-05
#> Iteration: 98 ELBO: -244.7566 Diff params: 0.0001188654 Diff ELBO: 7.736544e-06 
#> gradB : 2.8278e-12 gradD : 8.2706e-08 gradC : 2.4681e-05 gradM : 8.6882e-09 gradS : 1.8157e-05
#> gradB : 1.4443e-07 gradD : 3.2565e-11 gradC : 2.4667e-05 gradM : 8.8398e-09 gradS : 1.8157e-05
#> gradB : 3.3219e-05 gradD : 4.8691e-11 gradC : 1.6089e-12 gradM : 3.4102e-07 gradS : 2.2524e-05
#> gradB : 6.2804e-06 gradD : 2.1903e-08 gradC : 2.1964e-05 gradM : 9.9577e-09 gradS : 1.9445e-05
#> gradB : 7.1268e-06 gradD : 2.2213e-08 gradC : 3.0302e-05 gradM : 4.5679e-08 gradS : 1.2196e-08
#> Iteration: 99 ELBO: -244.7566 Diff params: 0.000124561 Diff ELBO: 1.950989e-06 
#> gradB : 1.7579e-11 gradD : 4.9187e-08 gradC : 3.4566e-05 gradM : 5.6209e-08 gradS : 8.0550e-09
#> gradB : 7.1324e-08 gradD : 2.1535e-11 gradC : 3.4559e-05 gradM : 5.6254e-08 gradS : 8.0936e-09
#> gradB : 3.7089e-05 gradD : 4.6723e-12 gradC : 2.1809e-12 gradM : 2.5114e-07 gradS : 4.7200e-07
#> gradB : 9.2666e-06 gradD : 2.8824e-09 gradC : 1.4122e-05 gradM : 1.7496e-09 gradS : 1.9054e-07
#> gradB : 6.2533e-06 gradD : 9.7311e-09 gradC : 1.7761e-05 gradM : 2.1140e-09 gradS : 9.8499e-08
#> Iteration: 100 ELBO: -244.7565 Diff params: 9.339711e-05 Diff ELBO: 4.698947e-06 
#> gradB : 2.1694e-12 gradD : 5.0828e-08 gradC : 2.3733e-05 gradM : 1.5085e-08 gradS : 8.0936e-08
#> gradB : 9.0914e-08 gradD : 1.8452e-11 gradC : 2.3722e-05 gradM : 1.5133e-08 gradS : 8.0914e-08
#> gradB : 3.0667e-05 gradD : 1.3466e-11 gradC : 1.9299e-11 gradM : 2.4467e-07 gradS : 6.0513e-07
#> gradB : 6.4541e-06 gradD : 3.3148e-09 gradC : 1.5249e-05 gradM : 3.9593e-09 gradS : 2.6346e-07
#> gradB : 4.3000e-06 gradD : 6.6187e-09 gradC : 1.9948e-05 gradM : 6.1334e-09 gradS : 5.7090e-08
#> Iteration: 101 ELBO: -244.7565 Diff params: 0.0001006874 Diff ELBO: 3.781189e-06 
#> gradB : 3.2530e-13 gradD : 3.7961e-08 gradC : 2.5438e-05 gradM : 2.3499e-08 gradS : 6.1345e-08
#> gradB : 6.5829e-08 gradD : 1.5323e-11 gradC : 2.5429e-05 gradM : 2.3567e-08 gradS : 6.1335e-08
#> gradB : 3.0555e-05 gradD : 3.0911e-11 gradC : 1.2074e-10 gradM : 2.2563e-07 gradS : 2.6820e-07
#> gradB : 1.3167e-05 gradD : 1.4992e-08 gradC : 5.4431e-06 gradM : 2.6160e-08 gradS : 1.3391e-07
#> gradB : 1.1417e-05 gradD : 2.3553e-08 gradC : 6.7272e-06 gradM : 2.4684e-08 gradS : 1.3528e-08
#> Iteration: 102 ELBO: -244.7565 Diff params: 7.810599e-05 Diff ELBO: 3.50084e-06 
#> gradB : 3.7801e-12 gradD : 1.1147e-07 gradC : 1.2419e-05 gradM : 7.4963e-09 gradS : 4.2614e-09
#> gradB : 1.8838e-07 gradD : 4.7205e-11 gradC : 1.2406e-05 gradM : 7.4545e-09 gradS : 4.2306e-09
#> gradB : 2.0375e-05 gradD : 1.9782e-11 gradC : 2.0148e-12 gradM : 2.2398e-07 gradS : 1.9782e-07
#> gradB : 2.8243e-06 gradD : 1.4888e-11 gradC : 1.1549e-05 gradM : 1.4323e-09 gradS : 4.0981e-08
#> gradB : 8.0378e-06 gradD : 3.5439e-09 gradC : 6.5027e-06 gradM : 2.1133e-08 gradS : 5.4184e-06
#> Iteration: 103 ELBO: -244.7565 Diff params: 9.617767e-05 Diff ELBO: 8.195128e-06 
#> gradB : 1.1503e-09 gradD : 2.1835e-08 gradC : 8.9341e-06 gradM : 1.7434e-08 gradS : 5.2939e-06
#> gradB : 3.1234e-08 gradD : 9.7562e-12 gradC : 8.9312e-06 gradM : 1.7489e-08 gradS : 5.2938e-06
#> gradB : 1.0171e-05 gradD : 7.7586e-12 gradC : 4.0108e-12 gradM : 1.1172e-07 gradS : 5.7838e-06
#> gradB : 2.6875e-06 gradD : 2.3171e-09 gradC : 7.7171e-06 gradM : 3.6474e-09 gradS : 5.4040e-06
#> gradB : 5.0658e-06 gradD : 2.3039e-09 gradC : 1.6813e-05 gradM : 2.5109e-08 gradS : 7.5436e-08
#> Iteration: 104 ELBO: -244.7565 Diff params: 0.0001101782 Diff ELBO: 1.880563e-06 
#> gradB : 1.2131e-12 gradD : 1.5954e-08 gradC : 1.9750e-05 gradM : 2.7912e-08 gradS : 6.4915e-08
#> gradB : 2.8622e-08 gradD : 8.4938e-11 gradC : 1.9745e-05 gradM : 2.7947e-08 gradS : 6.4901e-08
#> gradB : 2.2009e-05 gradD : 9.8694e-11 gradC : 3.2825e-10 gradM : 1.6988e-07 gradS : 2.3659e-07
#> gradB : 4.6590e-06 gradD : 2.5822e-09 gradC : 7.7922e-06 gradM : 6.4440e-11 gradS : 8.3555e-08
#> gradB : 4.5184e-06 gradD : 2.5877e-09 gradC : 8.1906e-06 gradM : 7.7043e-11 gradS : 1.3689e-08
#> Iteration: 105 ELBO: -244.7565 Diff params: 7.491869e-05 Diff ELBO: 4.411841e-07 
#> gradB : 1.6256e-12 gradD : 2.5096e-08 gradC : 1.1645e-05 gradM : 6.7425e-09 gradS : 1.6026e-08
#> gradB : 4.2777e-08 gradD : 1.0410e-11 gradC : 1.1640e-05 gradM : 6.7909e-09 gradS : 1.6048e-08
#> gradB : 1.4739e-05 gradD : 5.7417e-12 gradC : 5.5444e-12 gradM : 1.2253e-07 gradS : 1.2353e-07
#> gradB : 3.0233e-06 gradD : 3.5084e-09 gradC : 7.7880e-06 gradM : 2.0057e-09 gradS : 3.3745e-08
#> gradB : 1.6431e-06 gradD : 7.3090e-09 gradC : 1.0036e-05 gradM : 2.6062e-09 gradS : 1.7663e-08
#> Iteration: 106 ELBO: -244.7565 Diff params: 7.193633e-05 Diff ELBO: 1.529941e-06 
#> gradB : 3.6909e-11 gradD : 2.1221e-08 gradC : 1.2102e-05 gradM : 8.2637e-09 gradS : 1.8871e-08
#> gradB : 3.7583e-08 gradD : 9.0025e-12 gradC : 1.2098e-05 gradM : 8.2987e-09 gradS : 1.8867e-08
#> gradB : 1.4349e-05 gradD : 1.5557e-11 gradC : 5.1317e-11 gradM : 1.1824e-07 gradS : 1.5531e-07
#> gradB : 2.8373e-06 gradD : 2.4524e-09 gradC : 7.2835e-06 gradM : 1.5634e-09 gradS : 5.4589e-08
#> gradB : 1.9779e-06 gradD : 5.4187e-09 gradC : 8.0782e-06 gradM : 2.1029e-09 gradS : 2.7264e-07
#> Iteration: 107 ELBO: -244.7565 Diff params: 6.975271e-05 Diff ELBO: 3.13165e-06 
#> gradB : 7.9858e-13 gradD : 2.0962e-08 gradC : 1.0368e-05 gradM : 8.8670e-09 gradS : 2.7250e-07
#> gradB : 3.4947e-08 gradD : 9.0781e-12 gradC : 1.0363e-05 gradM : 8.9030e-09 gradS : 2.7249e-07
#> gradB : 1.4253e-05 gradD : 1.1625e-11 gradC : 3.8876e-11 gradM : 9.6969e-08 gradS : 4.7643e-07
#> gradB : 2.6707e-06 gradD : 2.3066e-09 gradC : 6.4943e-06 gradM : 2.3636e-09 gradS : 3.3943e-07
#> gradB : 4.0895e-06 gradD : 6.0489e-09 gradC : 3.1782e-06 gradM : 3.2668e-09 gradS : 8.4040e-06
#> Iteration: 108 ELBO: -244.7565 Diff params: 7.780179e-05 Diff ELBO: 4.121824e-06 
#> gradB : 1.5966e-12 gradD : 3.3979e-08 gradC : 5.4315e-06 gradM : 6.1919e-09 gradS : 8.2074e-06
#> gradB : 5.8206e-08 gradD : 1.4032e-11 gradC : 5.4270e-06 gradM : 6.2205e-09 gradS : 8.2075e-06
#> gradB : 8.4272e-06 gradD : 6.0525e-12 gradC : 4.7892e-12 gradM : 5.7379e-08 gradS : 9.7140e-06
#> gradB : 2.7427e-06 gradD : 2.3815e-12 gradC : 2.1505e-06 gradM : 3.0899e-09 gradS : 9.0429e-06
#> gradB : 1.0185e-06 gradD : 5.1995e-12 gradC : 7.0434e-06 gradM : 3.8235e-09 gradS : 3.0866e-09
#> Iteration: 109 ELBO: -244.7565 Diff params: 6.9768e-05 Diff ELBO: 5.857358e-07 
#> gradB : 7.4434e-12 gradD : 2.6788e-09 gradC : 8.4293e-06 gradM : 8.4355e-09 gradS : 3.0493e-09
#> gradB : 4.5557e-09 gradD : 2.5709e-13 gradC : 8.4283e-06 gradM : 8.4619e-09 gradS : 3.0522e-09
#> gradB : 8.5421e-06 gradD : 6.1077e-12 gradC : 9.5609e-11 gradM : 7.6028e-08 gradS : 1.0529e-07
#> gradB : 1.8179e-06 gradD : 6.7303e-09 gradC : 5.3769e-06 gradM : 8.3288e-09 gradS : 3.3565e-08
#> gradB : 1.0776e-06 gradD : 1.4461e-08 gradC : 1.3449e-05 gradM : 1.9876e-08 gradS : 8.4149e-06
#> Iteration: 110 ELBO: -244.7565 Diff params: 6.035515e-05 Diff ELBO: 2.350072e-06 
#> gradB : 2.8330e-13 gradD : 2.3067e-08 gradC : 1.4526e-05 gradM : 2.1808e-08 gradS : 8.4774e-06
#> gradB : 4.1717e-08 gradD : 8.1452e-12 gradC : 1.4522e-05 gradM : 2.1830e-08 gradS : 8.4773e-06
#> gradB : 1.5116e-05 gradD : 1.4910e-11 gradC : 5.1531e-11 gradM : 1.1922e-07 gradS : 6.2784e-06
#> gradB : 4.4427e-06 gradD : 9.2758e-13 gradC : 4.6263e-06 gradM : 2.0153e-09 gradS : 7.0861e-06
#> gradB : 4.4916e-06 gradD : 4.7400e-11 gradC : 5.7730e-06 gradM : 1.5303e-09 gradS : 1.4176e-05
#> Iteration: 111 ELBO: -244.7565 Diff params: 5.929699e-05 Diff ELBO: 3.270784e-06 
#> gradB : 1.4703e-11 gradD : 8.6219e-09 gradC : 8.1365e-06 gradM : 5.1446e-09 gradS : 1.4358e-05
#> gradB : 1.4002e-08 gradD : 3.9305e-12 gradC : 8.1342e-06 gradM : 5.1885e-09 gradS : 1.4358e-05
#> gradB : 1.0400e-05 gradD : 6.2472e-12 gradC : 7.7865e-11 gradM : 8.8326e-08 gradS : 1.2153e-05
#> gradB : 2.9389e-06 gradD : 4.0944e-09 gradC : 4.7217e-06 gradM : 2.6730e-09 gradS : 1.3308e-05
#> gradB : 4.4849e-06 gradD : 4.0840e-09 gradC : 1.4415e-06 gradM : 5.9652e-09 gradS : 1.3724e-09
#> Iteration: 112 ELBO: -244.7565 Diff params: 5.801238e-05 Diff ELBO: 8.079743e-07 
#> gradB : 4.5712e-13 gradD : 2.8460e-08 gradC : 3.0825e-06 gradM : 2.8507e-09 gradS : 1.1678e-09
#> gradB : 4.8382e-08 gradD : 1.1992e-11 gradC : 3.0783e-06 gradM : 2.8800e-09 gradS : 1.1869e-09
#> gradB : 5.5375e-06 gradD : 3.7823e-12 gradC : 2.5367e-13 gradM : 4.9314e-08 gradS : 3.3267e-08
#> gradB : 1.3052e-06 gradD : 2.2063e-10 gradC : 2.7389e-06 gradM : 1.6463e-10 gradS : 6.5242e-09
#> gradB : 2.6541e-07 gradD : 1.8815e-09 gradC : 7.0078e-06 gradM : 7.0922e-09 gradS : 8.8541e-06
#> Iteration: 113 ELBO: -244.7565 Diff params: 4.124491e-05 Diff ELBO: 2.071994e-06 
#> gradB : 2.5929e-10 gradD : 4.6829e-09 gradC : 7.6895e-06 gradM : 1.0009e-08 gradS : 8.9006e-06
#> gradB : 1.1004e-08 gradD : 2.0034e-12 gradC : 7.6885e-06 gradM : 1.0037e-08 gradS : 8.9006e-06
#> gradB : 7.5175e-06 gradD : 3.1793e-12 gradC : 8.5152e-10 gradM : 5.9599e-08 gradS : 7.2229e-06
#> gradB : 1.8574e-06 gradD : 5.6706e-09 gradC : 4.1349e-06 gradM : 8.3375e-09 gradS : 7.9065e-06
#> gradB : 2.8116e-06 gradD : 6.8398e-09 gradC : 2.0733e-06 gradM : 1.2832e-08 gradS : 1.1778e-08
#> Iteration: 114 ELBO: -244.7565 Diff params: 5.848539e-05 Diff ELBO: 1.587543e-06 
#> gradB : 2.7080e-12 gradD : 3.0024e-08 gradC : 3.4878e-06 gradM : 8.1673e-09 gradS : 1.4367e-08
#> gradB : 5.3863e-08 gradD : 1.0631e-11 gradC : 3.4849e-06 gradM : 8.1571e-09 gradS : 1.4369e-08
#> gradB : 4.8442e-06 gradD : 1.7515e-12 gradC : 2.1756e-10 gradM : 4.9759e-08 gradS : 3.5225e-08
#> gradB : 1.7429e-06 gradD : 5.1587e-13 gradC : 1.2437e-06 gradM : 3.2816e-09 gradS : 1.6851e-08
#> gradB : 9.2080e-07 gradD : 2.1536e-10 gradC : 2.1618e-06 gradM : 2.7482e-09 gradS : 1.1921e-07
#> Iteration: 115 ELBO: -244.7565 Diff params: 3.467324e-05 Diff ELBO: 1.787948e-06 
#> gradB : 1.0444e-11 gradD : 3.5112e-09 gradC : 2.9108e-06 gradM : 2.4122e-09 gradS : 1.2885e-07
#> gradB : 6.1988e-09 gradD : 3.0842e-13 gradC : 2.9098e-06 gradM : 2.4321e-09 gradS : 1.2884e-07
#> gradB : 4.0004e-06 gradD : 2.8143e-11 gradC : 3.5890e-13 gradM : 5.0058e-08 gradS : 3.4999e-08
#> gradB : 1.0059e-06 gradD : 1.4831e-11 gradC : 1.5702e-06 gradM : 2.2277e-09 gradS : 7.7018e-08
#> gradB : 4.1653e-07 gradD : 5.1741e-10 gradC : 8.4512e-06 gradM : 1.1925e-08 gradS : 3.4006e-05
#> Iteration: 116 ELBO: -244.7565 Diff params: 6.997344e-05 Diff ELBO: 1.374291e-06 
#> gradB : 1.7989e-10 gradD : 6.7119e-10 gradC : 8.4751e-06 gradM : 1.1172e-08 gradS : 3.3979e-05
#> gradB : 2.9889e-12 gradD : 4.3178e-11 gradC : 8.4750e-06 gradM : 1.1181e-08 gradS : 3.3978e-05
#> gradB : 6.2930e-06 gradD : 3.7472e-11 gradC : 4.2891e-10 gradM : 6.9892e-08 gradS : 3.0454e-05
#> gradB : 2.7436e-06 gradD : 8.8677e-09 gradC : 3.7254e-06 gradM : 1.8267e-08 gradS : 3.1669e-05
#> gradB : 6.0940e-06 gradD : 8.7707e-09 gradC : 3.1772e-07 gradM : 4.8533e-08 gradS : 1.4564e-08
#> Iteration: 117 ELBO: -244.7565 Diff params: 6.982992e-05 Diff ELBO: 1.968106e-06 
#> gradB : 6.5925e-12 gradD : 5.0984e-08 gradC : 1.1515e-06 gradM : 2.1835e-08 gradS : 1.7421e-08
#> gradB : 9.4544e-08 gradD : 1.5801e-11 gradC : 1.1477e-06 gradM : 2.1795e-08 gradS : 1.7414e-08
#> gradB : 2.6411e-06 gradD : 5.7174e-13 gradC : 4.2996e-11 gradM : 4.4720e-08 gradS : 1.9269e-08
#> gradB : 1.7736e-06 gradD : 9.9782e-13 gradC : 1.3761e-07 gradM : 2.4043e-08 gradS : 1.7722e-08
#> gradB : 1.3460e-06 gradD : 1.6071e-10 gradC : 2.9355e-07 gradM : 2.1872e-08 gradS : 1.4411e-07
#> Iteration: 118 ELBO: -244.7565 Diff params: 4.950484e-05 Diff ELBO: 9.297677e-07 
#> gradB : 4.9929e-13 gradD : 5.1233e-09 gradC : 7.6043e-07 gradM : 1.5367e-08 gradS : 1.5857e-07
#> gradB : 8.7807e-09 gradD : 2.1035e-12 gradC : 7.5925e-07 gradM : 1.5378e-08 gradS : 1.5856e-07
#> gradB : 1.7117e-06 gradD : 6.6676e-12 gradC : 5.0070e-12 gradM : 4.3285e-08 gradS : 9.6329e-08
#> gradB : 1.0722e-06 gradD : 4.4830e-12 gradC : 1.2326e-07 gradM : 2.4160e-08 gradS : 1.1445e-07
#> gradB : 9.0287e-07 gradD : 1.2302e-10 gradC : 6.5007e-08 gradM : 3.1539e-08 gradS : 3.8098e-07
#> Iteration: 119 ELBO: -244.7565 Diff params: 4.005017e-05 Diff ELBO: 1.156926e-06 
#> gradB : 3.8587e-13 gradD : 3.8170e-09 gradC : 3.0204e-07 gradM : 2.3180e-08 gradS : 3.6189e-07
#> gradB : 6.5672e-09 gradD : 1.5648e-12 gradC : 3.0080e-07 gradM : 2.3192e-08 gradS : 3.6188e-07
#> gradB : 9.8569e-07 gradD : 3.7535e-12 gradC : 3.6711e-10 gradM : 3.9832e-08 gradS : 4.1736e-07
#> gradB : 9.4774e-07 gradD : 3.6438e-12 gradC : 1.0505e-09 gradM : 3.8416e-08 gradS : 4.1495e-07
#> gradB : 1.4209e-06 gradD : 2.0165e-11 gradC : 1.7256e-07 gradM : 3.1911e-08 gradS : 1.4420e-07
#> Iteration: 120 ELBO: -244.7565 Diff params: 8.650383e-05 Diff ELBO: 1.043437e-06 
#> gradB : 4.2403e-12 gradD : 3.9855e-09 gradC : 2.8140e-07 gradM : 2.2257e-08 gradS : 1.4772e-07
#> gradB : 6.5859e-09 gradD : 1.7461e-12 gradC : 2.8175e-07 gradM : 2.2271e-08 gradS : 1.4772e-07
#> gradB : 1.4189e-07 gradD : 8.1928e-12 gradC : 2.2673e-11 gradM : 4.0554e-08 gradS : 1.2977e-07
#> gradB : 1.2765e-07 gradD : 8.1618e-12 gradC : 7.6917e-10 gradM : 3.9124e-08 gradS : 1.3016e-07
#> gradB : 5.4084e-08 gradD : 8.7048e-12 gradC : 4.1206e-08 gradM : 4.0956e-08 gradS : 6.6309e-09
#> Iteration: 121 ELBO: -244.7565 Diff params: 6.875695e-05 Diff ELBO: 1.037514e-07 
#> gradB : 2.8760e-12 gradD : 2.4684e-10 gradC : 5.9729e-08 gradM : 3.8171e-08 gradS : 6.1955e-09
#> gradB : 4.3996e-10 gradD : 4.5491e-13 gradC : 5.9564e-08 gradM : 3.8176e-08 gradS : 6.1955e-09
#> gradB : 1.9471e-07 gradD : 4.8034e-14 gradC : 1.0950e-10 gradM : 4.0354e-08 gradS : 8.5041e-09
#> gradB : 1.6015e-07 gradD : 5.0547e-14 gradC : 2.7334e-09 gradM : 3.7096e-08 gradS : 8.1171e-09
#> gradB : 1.1962e-07 gradD : 1.0223e-11 gradC : 9.0752e-08 gradM : 3.0116e-08 gradS : 6.4222e-07
#> Iteration: 122 ELBO: -244.7565 Diff params: 1.907447e-05 Diff ELBO: 2.300741e-07 
#> gradB : 4.2815e-11 gradD : 3.5815e-10 gradC : 1.3394e-07 gradM : 2.7703e-08 gradS : 6.4901e-07
#> gradB : 8.9725e-10 gradD : 7.6726e-13 gradC : 1.3396e-07 gradM : 2.7709e-08 gradS : 6.4901e-07
#> gradB : 1.1571e-07 gradD : 1.7018e-12 gradC : 6.6010e-10 gradM : 4.3358e-08 gradS : 5.8685e-07
#> gradB : 1.4408e-08 gradD : 2.0406e-12 gradC : 1.1848e-07 gradM : 2.4077e-08 gradS : 6.3013e-07
#> gradB : 6.9060e-09 gradD : 1.9442e-12 gradC : 3.9953e-08 gradM : 3.0177e-08 gradS : 8.0939e-10
#> Iteration: 123 ELBO: -244.7565 Diff params: 1.834021e-05 Diff ELBO: 3.92501e-08 
#> gradB : 1.5144e-10 gradD : 1.5917e-11 gradC : 4.6940e-08 gradM : 2.9877e-08 gradS : 7.9757e-10
#> gradB : 2.9038e-10 gradD : 1.1869e-13 gradC : 4.6908e-08 gradM : 2.9879e-08 gradS : 7.9757e-10
#> gradB : 1.3365e-07 gradD : 9.1212e-13 gradC : 2.1115e-10 gradM : 3.6320e-08 gradS : 1.6417e-09
#> gradB : 1.3365e-07 gradD : 9.1212e-13 gradC : 2.1114e-10 gradM : 3.6320e-08 gradS : 1.6417e-09
#> gradB : 9.1487e-08 gradD : 2.4782e-12 gradC : 3.9230e-09 gradM : 3.6360e-08 gradS : 1.3875e-10
#> Iteration: 124 ELBO: -244.7565 Diff params: 7.030805e-06 Diff ELBO: 3.223179e-07 
#> gradB : 1.1341e-10 gradD : 2.7082e-10 gradC : 2.0321e-08 gradM : 3.3543e-08 gradS : 1.8281e-10
#> gradB : 9.7736e-10 gradD : 5.8059e-13 gradC : 2.0236e-08 gradM : 3.3549e-08 gradS : 1.8272e-10
#> gradB : 5.3742e-08 gradD : 3.7120e-13 gradC : 1.3580e-09 gradM : 3.6312e-08 gradS : 6.7005e-11
#> gradB : 5.3739e-08 gradD : 3.7119e-13 gradC : 1.3585e-09 gradM : 3.6312e-08 gradS : 6.6996e-11
#> gradB : 4.3611e-08 gradD : 3.4936e-12 gradC : 2.3806e-08 gradM : 3.2082e-08 gradS : 2.6246e-07
#> Iteration: 125 ELBO: -244.7565 Diff params: 4.977262e-06 Diff ELBO: 3.287857e-07 
#> gradB : 1.3597e-11 gradD : 1.0959e-10 gradC : 3.7703e-08 gradM : 3.0752e-08 gradS : 2.6463e-07
#> gradB : 2.2088e-10 gradD : 2.3310e-13 gradC : 3.7708e-08 gradM : 3.0755e-08 gradS : 2.6462e-07
#> gradB : 5.0179e-08 gradD : 7.1598e-13 gradC : 7.6805e-10 gradM : 3.9695e-08 gradS : 2.4256e-07
#> gradB : 1.3349e-08 gradD : 8.1579e-13 gradC : 1.3956e-08 gradM : 3.1961e-08 gradS : 2.5327e-07
#> gradB : 2.1740e-08 gradD : 2.9828e-12 gradC : 3.5439e-09 gradM : 3.6421e-08 gradS : 1.3037e-09
#> Iteration: 126 ELBO: -244.7565 Diff params: 5.362574e-06 Diff ELBO: 2.759887e-08 
#> gradB : 7.5882e-10 gradD : 8.3325e-11 gradC : 7.7862e-09 gradM : 3.4928e-08 gradS : 1.1797e-09
#> gradB : 1.2040e-09 gradD : 1.6720e-13 gradC : 7.7508e-09 gradM : 3.4932e-08 gradS : 1.1797e-09
#> gradB : 3.4579e-08 gradD : 8.9357e-13 gradC : 1.8505e-11 gradM : 3.6396e-08 gradS : 1.3004e-09
#> gradB : 3.4572e-08 gradD : 8.9355e-13 gradC : 1.8378e-11 gradM : 3.6395e-08 gradS : 1.3002e-09
#> gradB : 2.6940e-08 gradD : 3.6994e-12 gradC : 6.2133e-10 gradM : 3.5797e-08 gradS : 1.1366e-09
#> Iteration: 127 ELBO: -244.7565 Diff params: 2.514834e-06 Diff ELBO: 1.938846e-07 
#> gradB : 1.2131e-09 gradD : 8.6986e-11 gradC : 4.9224e-09 gradM : 3.4400e-08 gradS : 1.2989e-09
#> gradB : 1.8127e-09 gradD : 1.7128e-13 gradC : 4.9065e-09 gradM : 3.4403e-08 gradS : 1.2988e-09
#> gradB : 1.2429e-08 gradD : 1.0296e-13 gradC : 4.0311e-10 gradM : 3.7137e-08 gradS : 8.5262e-10
#> gradB : 1.0528e-08 gradD : 1.0658e-13 gradC : 4.3507e-10 gradM : 3.6563e-08 gradS : 9.0002e-10
#> gradB : 5.9315e-09 gradD : 1.0981e-12 gradC : 1.1978e-08 gradM : 3.5731e-08 gradS : 2.6462e-08
#> Iteration: 128 ELBO: -244.7565 Diff params: 1.986419e-05 Diff ELBO: 9.877184e-08 
#> gradB : 2.5144e-10 gradD : 8.0372e-12 gradC : 1.3684e-08 gradM : 3.5171e-08 gradS : 2.6591e-08
#> gradB : 3.8390e-10 gradD : 1.1610e-12 gradC : 1.3678e-08 gradM : 3.5172e-08 gradS : 2.6591e-08
#> gradB : 3.1140e-08 gradD : 8.1542e-13 gradC : 1.1855e-10 gradM : 4.0409e-08 gradS : 2.3944e-08
#> gradB : 7.1923e-09 gradD : 1.0748e-12 gradC : 1.7786e-08 gradM : 3.2573e-08 gradS : 2.5860e-08
#> gradB : 9.5034e-09 gradD : 1.2885e-12 gradC : 6.7837e-09 gradM : 3.3729e-08 gradS : 4.5676e-09
#> Iteration: 129 ELBO: -244.7565 Diff params: 1.500679e-05 Diff ELBO: 9.688955e-08 
#> gradB : 2.7189e-10 gradD : 1.5052e-11 gradC : 1.0407e-08 gradM : 3.3229e-08 gradS : 4.6727e-09
#> gradB : 4.6139e-10 gradD : 1.0586e-14 gradC : 1.0401e-08 gradM : 3.3231e-08 gradS : 4.6727e-09
#> gradB : 2.6085e-08 gradD : 4.4831e-14 gradC : 1.1097e-10 gradM : 3.7622e-08 gradS : 4.3168e-09
#> gradB : 1.4239e-08 gradD : 2.1186e-14 gradC : 2.3024e-09 gradM : 3.4625e-08 gradS : 4.1973e-09
#> gradB : 2.4350e-08 gradD : 6.9201e-13 gradC : 7.0286e-10 gradM : 3.4502e-08 gradS : 1.8559e-10
#> Iteration: 130 ELBO: -244.7565 Diff params: 2.874167e-06 Diff ELBO: 9.494136e-08 
#> gradB : 6.5455e-10 gradD : 3.7454e-11 gradC : 2.0446e-09 gradM : 3.3702e-08 gradS : 1.8344e-10
#> gradB : 1.0475e-09 gradD : 9.3020e-14 gradC : 2.0445e-09 gradM : 3.3704e-08 gradS : 1.8343e-10
#> gradB : 5.5948e-09 gradD : 1.1227e-13 gradC : 1.9796e-10 gradM : 3.5525e-08 gradS : 2.0061e-10
#> gradB : 5.5948e-09 gradD : 1.1227e-13 gradC : 1.9796e-10 gradM : 3.5525e-08 gradS : 2.0061e-10
#> gradB : 5.5850e-09 gradD : 1.1441e-13 gradC : 1.7345e-10 gradM : 3.5558e-08 gradS : 4.4042e-11
#> Iteration: 131 ELBO: -244.7565 Diff params: 8.03362e-07 Diff ELBO: 6.885728e-10 
#> gradB : 1.9650e-10 gradD : 1.6586e-11 gradC : 7.8764e-10 gradM : 3.4797e-08 gradS : 4.0365e-11
#> gradB : 2.9358e-10 gradD : 3.1167e-14 gradC : 7.8537e-10 gradM : 3.4799e-08 gradS : 4.0368e-11
#> gradB : 2.9948e-09 gradD : 4.1116e-14 gradC : 2.8095e-10 gradM : 3.6011e-08 gradS : 5.5255e-11
#> gradB : 2.9911e-09 gradD : 4.1117e-14 gradC : 2.8012e-10 gradM : 3.6009e-08 gradS : 5.5223e-11
#> gradB : 3.0960e-09 gradD : 4.7092e-13 gradC : 3.7021e-10 gradM : 3.5957e-08 gradS : 6.1295e-11
#> Iteration: 132 ELBO: -244.7565 Diff params: 5.576747e-07 Diff ELBO: 7.520134e-08 
#> gradB : 1.2781e-10 gradD : 1.1385e-11 gradC : 4.1009e-10 gradM : 3.5421e-08 gradS : 6.0012e-11
#> gradB : 1.9532e-10 gradD : 1.6055e-14 gradC : 4.0900e-10 gradM : 3.5422e-08 gradS : 6.0011e-11
#> gradB : 1.9234e-09 gradD : 1.7465e-13 gradC : 2.3183e-11 gradM : 3.5439e-08 gradS : 5.9430e-11
#> gradB : 1.9234e-09 gradD : 1.7465e-13 gradC : 2.3183e-11 gradM : 3.5439e-08 gradS : 5.9430e-11
#> gradB : 1.9355e-09 gradD : 4.5341e-13 gradC : 3.1801e-10 gradM : 3.5678e-08 gradS : 1.9567e-09
#> Iteration: 133 ELBO: -244.7565 Diff params: 2.313947e-06 Diff ELBO: 3.821063e-08 
#> gradB : 2.9672e-11 gradD : 7.9958e-12 gradC : 6.5681e-10 gradM : 3.5188e-08 gradS : 1.9289e-09
#> gradB : 6.4754e-11 gradD : 8.0817e-15 gradC : 6.5748e-10 gradM : 3.5189e-08 gradS : 1.9289e-09
#> gradB : 9.1882e-10 gradD : 1.1619e-14 gradC : 1.0487e-10 gradM : 3.6223e-08 gradS : 1.9766e-09
#> gradB : 6.9870e-10 gradD : 1.2888e-14 gradC : 9.3272e-11 gradM : 3.5972e-08 gradS : 1.9613e-09
#> gradB : 3.1118e-10 gradD : 7.7832e-14 gradC : 6.7453e-10 gradM : 3.4791e-08 gradS : 1.2437e-08
#> Iteration: 134 ELBO: -244.7565 Diff params: 1.932324e-06 Diff ELBO: 1.499072e-08 
#> gradB : 2.9555e-12 gradD : 4.6035e-13 gradC : 8.0639e-10 gradM : 3.4751e-08 gradS : 1.2454e-08
#> gradB : 3.8666e-12 gradD : 1.3442e-13 gradC : 8.0639e-10 gradM : 3.4752e-08 gradS : 1.2454e-08
#> gradB : 9.3581e-10 gradD : 1.2426e-13 gradC : 4.5251e-11 gradM : 3.5868e-08 gradS : 1.1834e-08
#> gradB : 9.2688e-10 gradD : 1.2443e-13 gradC : 4.5490e-11 gradM : 3.5858e-08 gradS : 1.1837e-08
#> gradB : 8.8113e-10 gradD : 1.8897e-13 gradC : 6.5569e-11 gradM : 3.5927e-08 gradS : 1.0283e-08
#> Iteration: 135 ELBO: -244.7565 Diff params: 2.980909e-07 Diff ELBO: 1.573986e-08 
#> gradB : 3.9318e-11 gradD : 3.2879e-12 gradC : 1.9920e-10 gradM : 3.5664e-08 gradS : 1.0370e-08
#> gradB : 6.1131e-11 gradD : 1.5760e-12 gradC : 1.9875e-10 gradM : 3.5664e-08 gradS : 1.0370e-08
#> gradB : 1.0377e-09 gradD : 1.2436e-12 gradC : 9.8984e-12 gradM : 3.5977e-08 gradS : 1.0217e-08
#> gradB : 1.0357e-09 gradD : 1.2436e-12 gradC : 9.8576e-12 gradM : 3.5975e-08 gradS : 1.0218e-08
#> gradB : 2.0394e-09 gradD : 1.1144e-12 gradC : 8.5245e-10 gradM : 3.6904e-08 gradS : 2.3984e-11
#> Iteration: 136 ELBO: -244.7565 Diff params: 9.324065e-07 Diff ELBO: 2.86702e-08 
#> gradB : 8.0855e-11 gradD : 5.2745e-12 gradC : 4.2866e-10 gradM : 3.6453e-08 gradS : 1.7631e-11
#> gradB : 1.1937e-10 gradD : 1.2018e-13 gradC : 4.2797e-10 gradM : 3.6454e-08 gradS : 1.7630e-11
#> gradB : 6.6062e-11 gradD : 1.0323e-13 gradC : 4.4242e-11 gradM : 3.5633e-08 gradS : 5.1870e-12
#> gradB : 6.6062e-11 gradD : 1.0323e-13 gradC : 4.4242e-11 gradM : 3.5633e-08 gradS : 5.1870e-12
#> gradB : 7.4821e-11 gradD : 1.0239e-13 gradC : 7.7241e-11 gradM : 3.5472e-08 gradS : 2.6983e-10
fit$elbo
#> [1] -244.7565
plot(fit$elboPath, type = "l", xlab = "Outer iteration", ylab = "ELBO")

```
