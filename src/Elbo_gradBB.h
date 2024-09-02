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
    arma::mat, double, double, double, double, double, double,
    arma::mat, arma::mat, arma::mat, arma::mat, arma::mat, arma::mat
>
Elbo_grad(const arma::mat & Y, const arma::mat & X, const arma::mat & R,
              const arma::mat & B, const arma::mat & D, const arma::mat & C, 
              const arma::mat & M, const arma::mat & S) {
              

    int n = Y.n_rows;
    int p = Y.n_cols;
    int q = M.n_cols;
    arma::vec XB = X * B;
    arma::vec XD = X * D;
    
    arma::mat mu = arma::mat(XB.memptr(), n, p, false, false);
    arma::mat nu = arma::mat(XD.memptr(), n, p, false, false);
    arma::vec vecY = arma::vectorise(Y);
    arma::vec vecR = arma::vectorise(R);
    arma::mat Z = mu + M * C.t();
    arma::vec vecZ = arma::vectorise(Z);
    arma::mat A = exp(Z + 0.5 * S * (C % C).t());
    arma::vec vecA = vectorise(A);
    arma::mat log_fact_Y = log_factorial_matrix(Y);
    arma::mat pi = 1./(1. + exp(-nu));
    arma::vec vecpi = vectorise(pi);
    arma::mat xi = ifelse_mat(Y, A, nu, R);
    arma::vec vecxi = vectorise(xi);
    
    double elbo1 = accu(xi % nu - ifelse_exp(nu));
    double elbo2 = - 0.5 * accu(M % M + S);
    //double elbo3 = accu(R % xi % (Y % (mu + M * C.t()) - A - log_fact_Y));
    double elbo3 = Elbo3(R, xi, mu, Y, M, C, A, log_fact_Y) ;
    double elbo4 = entropie_logis(xi);
    double elbo5 = 0.5 * accu(0.5 * log(S % S)) + n * q * 0.5;
    
   
    
    double objective = elbo1 + elbo2 + elbo3 + elbo4 + elbo5 ;
    
                        
    arma::mat gradB = GradB(vecY, X, vecR, vecxi, vecA);
    arma::mat gradD = X.t() * (vecR % (vecxi - vecpi));
    //arma::mat gradC = (R % xi % (Y - A)).t() * M - (R % xi % A).t() * S % C;
    arma::mat gradC = GradC(R, xi, Y, A, M, S, C);
    //arma::mat gradM = (R % xi % (Y - A) * C - M);
    arma::mat gradM = GradM(R, xi, Y, A, M, C);
    //arma::mat gradS = 0.5 * (1. / S - 1. - R % xi % A * (C % C));
    arma::mat gradS = GradS(R, xi, A, S, C);
    
    return std::make_tuple(
        xi, elbo1, elbo2, elbo3, elbo4, elbo5, objective,
        gradB, gradD, gradC, gradM, gradS, A
    );
}




inline std::tuple<
    arma::mat, double, double, double, double, double, double,
    arma::mat, arma::mat, arma::mat, arma::mat, arma::mat, arma::mat
>
Elbo_grad_LogS(const arma::mat & Y, const arma::mat & X, const arma::mat & R,
              const arma::mat & B, const arma::mat & D, const arma::mat & C, 
              const arma::mat & M, const arma::mat & logS
                ) {

    int n = Y.n_rows;
    int p = Y.n_cols;
    int q = M.n_cols;
    arma::mat S = exp(logS) ;
    auto [xi, elbo1, elbo2, elbo3, elbo4, elbo5, objective, gradB, gradD, gradC, gradM, gradS, A] = 
            Elbo_grad(Y, X, R, B, D, C, M, S);


    gradS = S % gradS;

     return std::make_tuple(
        xi, elbo1, elbo2, elbo3, elbo4, elbo5, objective,
        gradB, gradD, gradC, gradM, gradS, A
    );
}






