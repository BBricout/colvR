# Package index

## Core modelling

Fit, model selection, variance and confidence intervals.

- [`Miss.ZIPLNPCA()`](https://bbricout.github.io/colvR/reference/Miss.ZIPLNPCA.md)
  : ZI‑PLN-PCA with missing data (variational fit + imputation)
- [`select_model_bic()`](https://bbricout.github.io/colvR/reference/select_model_bic.md)
  : Model selection over candidate latent dimensions (BIC or ICL)
- [`V_theta()`](https://bbricout.github.io/colvR/reference/V_theta.md) :
  Variance-covariance matrix of theta
- [`IC()`](https://bbricout.github.io/colvR/reference/IC.md) : 95%
  interval for a theta 95% confidence interval for theta
- [`BIC()`](https://bbricout.github.io/colvR/reference/BIC.md) : BIC
  from the ELBO for PLN-PCA and ZIP-PLN-PCA models
- [`ICL()`](https://bbricout.github.io/colvR/reference/ICL.md) : ICL
  from the ELBO for PLN-PCA and ZIP-PLN-PCA models

## Initialisation

Helpers for initial parameters.

- [`Init()`](https://bbricout.github.io/colvR/reference/Init.md) :
  Initialize parameters for a PLN-PCA with missing data
- [`Init_ZIP()`](https://bbricout.github.io/colvR/reference/Init_ZIP.md)
  : Initialize parameters for the ZIP case (zero-inflated)
- [`Init_ZIP_q0()`](https://bbricout.github.io/colvR/reference/Init_ZIP_q0.md)
  : Init_ZIP_q0

## Optimisation & maths

Lower bounds, gradients, covariance utilities.

- [`Elbo_grad()`](https://bbricout.github.io/colvR/reference/Elbo_grad.md)
  : ELBO and parameter gradients (ZIP‑PLN variational objective)
- [`Elbo_grad_logS()`](https://bbricout.github.io/colvR/reference/Elbo_grad_logS.md)
  : Calculate the Elbo with log(S) instead of S and the gradients
- [`covmat()`](https://bbricout.github.io/colvR/reference/covmat.md) :
  Wrapper for \`lori::covmat\`

## Additional models & helpers

Complementary wrappers and step-wise routines.

- [`Miss.PLNPCA()`](https://bbricout.github.io/colvR/reference/Miss.PLNPCA.md)
  : PLN-PCA with missing data (variational fit + imputation)
- [`Miss.ZIPLNPCA.logS()`](https://bbricout.github.io/colvR/reference/Miss.ZIPLNPCA.logS.md)
  : ZI‑PLN-PCA with missing data using log(S) parametrization
- [`Miss.ZIPLNPCA_Steps()`](https://bbricout.github.io/colvR/reference/Miss.ZIPLNPCA_Steps.md)
  : ZI‑PLN-PCA (missing data) with parameterwise optimization steps
- [`Miss.ZIPLNPCA_VE()`](https://bbricout.github.io/colvR/reference/Miss.ZIPLNPCA_VE.md)
  : ZI‑PLN-PCA (missing data) — variational E‑step solver
- [`Predictions()`](https://bbricout.github.io/colvR/reference/Predictions.md)
  : Predictive simulations and intervals

## Utilities & simulation

Simulation tools and helpers.

- [`Simul()`](https://bbricout.github.io/colvR/reference/Simul.md) :
  Simulate zero-inflated Poisson log-normal PCA data

## Data

Datasets bundled with colvR.

- [`waterbird`](https://bbricout.github.io/colvR/reference/waterbird.md)
  : Waterbird count data
