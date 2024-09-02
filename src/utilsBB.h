#pragma once

#include "RcppArmadillo.h"
#include <Rcpp.h>

// Déclarations des fonctions


Rcpp::NumericVector MatrixToVector(const arma::mat & matrix);

Rcpp::NumericMatrix VectorToMatrix(const arma::vec & vector, int n, int p);

double log_factorial(double n);

arma::mat log_factorial_matrix(const arma::mat& Y);

arma::mat ifelse_mat(const arma::mat& Y, const arma::mat& A, const arma::mat& nu, const arma::mat& R);

arma::mat ifelse_exp(const arma::mat& nu);

double entropie_logis(arma::mat & xi);

arma::mat GradB(const arma::vec & vecY, const arma::mat & X, const arma::vec & vecR, const arma::vec & vecxi, const arma::vec & vecA);

arma::mat GradC(const arma::mat & R, const arma::mat & xi, const arma::mat & Y, const arma::mat & A,
		const arma::mat & M, const arma::mat & S, const arma::mat & C);

arma::mat GradM(const arma::mat & R, const arma::mat & xi, const arma::mat & Y, const arma::mat & A,
		const arma::mat & M, const arma::mat & C);

arma::mat GradS(const arma::mat & R, const arma::mat & xi, const arma::mat & A,
		const arma::mat & S, const arma::mat & C);

double Elbo3(const arma::mat & R, const arma::mat & xi, const arma::mat & mu,
		const arma::mat & Y, const arma::mat & M, const arma::mat & C,
		const arma::mat & A, const arma::mat & log_fact_Y);

