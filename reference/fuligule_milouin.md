# Fuligule milouin example dataset

Small example data for the vignette and examples.

## Usage

``` r
data(fuligule_milouin)
```

## Format

A named list with:

- Y:

  Count matrix of size \\n \times p\\.

- X:

  Covariate matrix of size \\n\*p \times d\\ (first column may be an
  intercept).

## Source

Internal.

## Details

Dimensions are kept small so that checks and vignettes run quickly.

## Examples

``` r
data(fuligule_milouin, package = "colvR")
str(fuligule_milouin, max.level = 1)
#> List of 2
#>  $ Y: num [1:419, 1:17] NA 0 NA NA NA 0 6 1 6 NA ...
#>   ..- attr(*, "dimnames")=List of 2
#>  $ X: num [1:7123, 1:8] 1 1 1 1 1 1 1 1 1 1 ...
#>   ..- attr(*, "dimnames")=List of 2
```
