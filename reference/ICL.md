# ICL from the ELBO for PLN-PCA and ZIP-PLN-PCA models

Integrated Completed Likelihood: the BIC minus the entropy of the
variational distribution of the latent variables (higher is better):
\$\$\mathrm{ICL} = \mathrm{BIC} - \mathcal{H}(W) - \mathcal{H}(U).\$\$

- \\\mathcal{H}(W)\\: entropy of the Gaussian variational distribution
  of \\W\\, \\\frac{1}{2}\sum\_{i,k} \log \sigma^2\_{ik} +
  \frac{nq}{2}\\1 + \log(2\pi)\\\\, where \\\sigma^2 = S\\ for the
  ZIP-PLN-PCA model (`S` is a variance) and \\\sigma^2 = S^2\\ for the
  PLN-PCA model (`S` is a standard deviation). Zero when `q = 0`.

- \\\mathcal{H}(U)\\: entropy of the Bernoulli variational distribution
  of \\U\\, \\-\sum\_{ij} \\\xi\_{ij}\log\xi\_{ij} +
  (1-\xi\_{ij})\log(1-\xi\_{ij})\\\\ (ZIP-PLN-PCA model only).

## Usage

``` r
ICL(Y, X, fit, q, model = c("ZIPLNPCA", "PLNPCA"))
```

## Arguments

- Y:

  Numeric `n x p` count matrix (NAs allowed).

- X:

  Design matrix, `(n*p) x d` (or `n x d` for a rowwise PLN-PCA design).

- fit:

  Fitted object containing `$elbo` (output of `Miss.ZIPLNPCA` or
  `Miss.PLNPCA`).

- q:

  Integer, size of the latent space.

- model:

  `"ZIPLNPCA"` (default) or `"PLNPCA"`.

## Value

A single numeric value, the ICL score.

## See also

[`BIC`](https://bbricout.github.io/colvR/reference/BIC.md)
