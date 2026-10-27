# colvR 0.2.0
## Breaking changes

* `Predictions.marginales()` has been removed: use `Predictions(type = "marginal")`.
* `select_model_bic()` now returns a list with the criterion values, the selected
  latent dimension (`q_star`) and the corresponding fit (`fit_star`), instead of a
  vector of BIC values.
* In `Miss.PLNPCA()`, `q` is now the third argument, as in `Miss.ZIPLNPCA()`.
* The example dataset `fuligule_milouin` has been replaced by `waterbird`, with
  anonymized sites and years and without the latitude covariate.

## New features

* `Predictions()` gains the arguments `type` (conditional or marginal intervals),
  `probs` (quantile levels), `output` (draws, latent variables and parameter draws
  in addition to the intervals), `cl` (parallel VE steps) and `verbose`.
* New function `ICL()`, the Integrated Completed Likelihood.
* `select_model_bic()` gains the arguments `criterion` (BIC or ICL) and `model`.
* `BIC()` gains the argument `model` (`"ZIPLNPCA"` or `"PLNPCA"`).

## Bug fixes

* `BIC()` now counts `2d` regression parameters for the ZIP-PLN-PCA model, and `d`
  or `pd` for the PLN-PCA model, depending on the design.
* `select_model_bic()` now works with non-contiguous values of `latents`.
* In the `Miss.ZIPLNPCA*()` functions, `pred$predicted` is now the ZIP expectation
  `xi * A`, consistent with `imputed`.
* `Predictions()` no longer fails when `q = 1`.

## Other changes

* All examples are now runnable, on small simulated data.
* New unit tests for the selection criteria, `select_model_bic()`, `Predictions()`
  and `Miss.ZIPLNPCA()`.
* All messages are now in English.

# colvR 0.2.1
## Changes
* Explanation of acronyms
* Removal of console output from Miss.ZIPLNPCA_Steps()
