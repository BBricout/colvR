# Waterbird count data

Counts of a waterbird species from a multi-year, multi-site monitoring
program. The data are incomplete: many site-year combinations were not
surveyed and are recorded as missing values. Counts also contain a large
proportion of zeros, which makes the data a suitable example for the
zero-inflated Poisson log-normal model implemented in the package. Sites
and years are anonymized.

## Usage

``` r
data(waterbird)
```

## Format

A named list with two elements:

- Y:

  Numeric count matrix with 419 rows (sites, `site_001` to `site_419`)
  and 17 columns (years, `year_01` to `year_17`). Unobserved site-year
  combinations are `NA`.

- X:

  Numeric design matrix with \\419 \times 17 = 7123\\ rows (site-year
  combinations, stacked year by year, i.e. by blocks of 419 sites) and 7
  columns: `intercept` (column of ones), and the standardized covariates
  `dist_coast` (distance to the coast), `dist_city` (distance to the
  nearest city), `elevation`, `year` (linear time trend),
  `spring_temp_anomaly` (spring temperature anomaly) and
  `agricultural_land` (percentage of agricultural land).

## Source

Anonymized data from a waterbird monitoring program.

## Examples

``` r
data(waterbird, package = "colvR")
str(waterbird, max.level = 1)
#> List of 2
#>  $ Y: num [1:419, 1:17] NA 0 NA NA NA 0 6 1 6 NA ...
#>   ..- attr(*, "dimnames")=List of 2
#>  $ X: num [1:7123, 1:7] 1 1 1 1 1 1 1 1 1 1 ...
#>   ..- attr(*, "dimnames")=List of 2
mean(is.na(waterbird$Y))                # proportion of missing entries
#> [1] 0.4443353
mean(waterbird$Y == 0, na.rm = TRUE)    # proportion of zeros
#> [1] 0.6705407
```
