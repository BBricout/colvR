#' Fuligule milouin example dataset
#'
#' Small example data for the vignette and examples.
#'
#' @format A named list with:
#' \describe{
#'   \item{Y}{Count matrix of size \eqn{n \times p}.}
#'   \item{X}{Covariate matrix of size \eqn{n*p \times d} (first column may be an intercept).}
#' }
#'
#' @details
#' Dimensions are kept small so that checks and vignettes run quickly.
#'
#' @source Internal.
#'
#' @examples
#' data(fuligule_milouin, package = "colvR")
#' str(fuligule_milouin, max.level = 1)
#'
#' @docType data
#' @usage data(fuligule_milouin)
#' @keywords datasets
"fuligule_milouin"
