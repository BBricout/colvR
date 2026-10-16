# Changelog

## colvR 0.2.0

### Breaking changes

- `Predictions.marginales()` has been removed: use
  `Predictions(type = "marginal")`.
- [`select_model_bic()`](https://bbricout.github.io/colvR/reference/select_model_bic.md)
  now returns a list with the criterion values, the selected latent
  dimension (`q_star`) and the corresponding fit (`fit_star`), instead
  of a vector of BIC values.
- In
  [`Miss.PLNPCA()`](https://bbricout.github.io/colvR/reference/Miss.PLNPCA.md),
  `q` is now the third argument, as in
  [`Miss.ZIPLNPCA()`](https://bbricout.github.io/colvR/reference/Miss.ZIPLNPCA.md).
- The example dataset `fuligule_milouin` has been replaced by
  `waterbird`, with anonymized sites and years and without the latitude
  covariate.

### New features

- [`Predictions()`](https://bbricout.github.io/colvR/reference/Predictions.md)
  gains the arguments `type` (conditional or marginal intervals),
  `probs` (quantile levels), `output` (draws, latent variables and
  parameter draws in addition to the intervals), `cl` (parallel VE
  steps) and `verbose`.
- New function
  [`ICL()`](https://bbricout.github.io/colvR/reference/ICL.md), the
  Integrated Completed Likelihood.
- [`select_model_bic()`](https://bbricout.github.io/colvR/reference/select_model_bic.md)
  gains the arguments `criterion` (BIC or ICL) and `model`.
- [`BIC()`](https://bbricout.github.io/colvR/reference/BIC.md) gains the
  argument `model` (`"ZIPLNPCA"` or `"PLNPCA"`).

### Bug fixes

- [`BIC()`](https://bbricout.github.io/colvR/reference/BIC.md) now
  counts `2d` regression parameters for the ZIP-PLN-PCA model, and `d`
  or `pd` for the PLN-PCA model, depending on the design.
- [`select_model_bic()`](https://bbricout.github.io/colvR/reference/select_model_bic.md)
  now works with non-contiguous values of `latents`.
- In the `Miss.ZIPLNPCA*()` functions, `pred$predicted` is now the ZIP
  expectation `xi * A`, consistent with `imputed`.
- [`Predictions()`](https://bbricout.github.io/colvR/reference/Predictions.md)
  no longer fails when `q = 1`.

### Other changes

- All examples are now runnable, on small simulated data.
- New unit tests for the selection criteria,
  [`select_model_bic()`](https://bbricout.github.io/colvR/reference/select_model_bic.md),
  [`Predictions()`](https://bbricout.github.io/colvR/reference/Predictions.md)
  and
  [`Miss.ZIPLNPCA()`](https://bbricout.github.io/colvR/reference/Miss.ZIPLNPCA.md).
- All messages are now in English.
