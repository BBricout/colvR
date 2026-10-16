# BIC from the ELBO for PLN-PCA and ZIP-PLN-PCA models

Computes a BIC-like score in which the log-likelihood is replaced by the
ELBO of the fitted model (higher is better): \$\$\mathrm{BIC} =
\mathrm{ELBO} - \frac{\log(n)}{2}\Big\[q\\p - (q-1)/2\\ + k\Big\],\$\$
where \\q\\p - (q-1)/2\\\\ is the number of free parameters in \\C\\
(identifiable up to a rotation) and \\k\\ the number of regression
parameters: \\2d\\ for the ZIP-PLN-PCA model (\\\beta\\ and \\\gamma\\),
\\d\\ for the PLN-PCA model with a vectorized design (\\(np) \times d\\)
and \\pd\\ with a rowwise design (\\n \times d\\). \\n\\ is the number
of rows of `Y` (sites), which are the independent units of the model.

## Usage

``` r
BIC(Y, X, fit, q, model = c("ZIPLNPCA", "PLNPCA"))
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

A single numeric value, the BIC score.

## See also

[`ICL`](https://bbricout.github.io/colvR/reference/ICL.md)
