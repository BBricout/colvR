#pragma once

#include "RcppArmadillo.h"
#include <cmath>
#include <iostream>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(nloptr)]]
// [[Rcpp::plugins(cpp11)]]

#include "nlopt_wrapper.h"
#include "packing.h"
#include "utils.h"
#include "utilsBB.h"



//--------------------------------------------------------------------------------------------------------------------
// Calcul de l'Elbo et des gradients


inline std::tuple<
    arma::mat, double, double, double, double,
    arma::mat, arma::mat, arma::mat
>
Elbo_grad_q0(const arma::mat & Y, const arma::mat & X, const arma::mat & R,
              const arma::mat & B, const arma::mat & D, double tolxi) {
              

    int n = Y.n_rows;
    int p = Y.n_cols;
    
    arma::vec XB = X * B;
    arma::vec XD = X * D;

    arma::mat mu = arma::reshape(XB, n, p);
    arma::mat nu = arma::reshape(XD, n, p);

    arma::vec vecY = arma::vectorise(Y);
    arma::vec vecR = arma::vectorise(R);


    arma::vec vecmu = arma::vectorise(mu);

    arma::mat A = exp(mu);

    arma::vec vecA = vectorise(A);

    arma::mat log_fact_Y = log_factorial_matrix(Y);

    arma::mat pi = 1./(1. + exp(-nu));

    arma::vec vecpi = vectorise(pi);
    arma::mat xi = ifelse_mat(Y, A, nu, R, tolxi);

    arma::vec vecxi = vectorise(xi);
    
    double elbo1 = accu(xi % nu - ifelse_exp(nu));


    //double elbo3 = accu(R % xi % (Y % (mu + M * C.t()) - A - log_fact_Y));
    double elbo3 = Elbo3_q0(R, xi, mu, Y, A, log_fact_Y) ;

    double elbo4 = entropie_logis(xi);


    
   
    
    double objective = elbo1 + elbo3 + elbo4 ;
    
                        
    arma::mat gradB = GradB(vecY, X, vecR, vecxi, vecA);
    arma::mat gradD = X.t() * (vecxi - vecpi);

    return std::make_tuple(
        xi, elbo1, elbo3, elbo4, objective,
        gradB, gradD, A
    );
}
