## Submission context

This is an update of a package already on CRAN (0.2.1). It fixes the
installation WARNING reported on r-devel-linux-x86_64-fedora-gcc (GCC 16):

  armadillo_bits/memory.hpp:125:14: warning: 'void free(void*)' called on
  pointer returned from a mismatched allocation function [-Wmismatched-new-delete]

The warning was triggered by an internal test function (`cpp_test_packing()`),
which has been removed. Installation was verified to be warning-free with
g++ 16.2.1 20261029 (Red Hat 16.2.1-3) on Fedora 44.

This update comes only shortly after 0.2.1 because it fixes the installation
WARNING on r-devel-linux-x86_64-fedora-gcc that CRAN asked us to correct
before 2026-12-04.

## Test environments

* local: Ubuntu 22.04, R 4.6.1
* Fedora 44, g++ 16.2.1, R 4.6.2 (installation only)
* win-builder: Windows Server 2022, R-devel
* GitHub Actions: Ubuntu (R release, devel and oldrel-1), macOS (R release),
  Windows (R release)

## R CMD check results

0 errors | 0 warnings | 0 notes