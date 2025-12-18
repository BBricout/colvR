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
              const arma::mat & M, const arma::mat & S, double tolxi) {
              

    int n = Y.n_rows;
    int p = Y.n_cols;
    int q = M.n_cols;
    arma::vec XB = X * B;
    arma::vec XD = X * D;

    arma::mat mu = arma::reshape(XB, n, p);
    arma::mat nu = arma::reshape(XD, n, p);

    arma::vec vecY = arma::vectorise(Y);
    arma::vec vecR = arma::vectorise(R);

    arma::mat Z = mu + M * C.t();

    arma::vec vecZ = arma::vectorise(Z);

    arma::mat A = exp(Z + 0.5 * S * (C % C).t());

    arma::vec vecA = vectorise(A);

    arma::mat log_fact_Y = log_factorial_matrix(Y);

    arma::mat pi = 1./(1. + exp(-nu));

    arma::vec vecpi = vectorise(pi);
    arma::mat xi = ifelse_mat(Y, A, nu, R, tolxi);

    arma::vec vecxi = vectorise(xi);
    
    double elbo1 = accu(xi % nu - ifelse_exp(nu));

    double elbo2 = - 0.5 * accu(M % M + S);

    //double elbo3 = accu(R % xi % (Y % (mu + M * C.t()) - A - log_fact_Y));
    double elbo3 = Elbo3(R, xi, mu, Y, M, C, A, log_fact_Y) ;

    double elbo4 = entropie_logis(xi);

    double elbo5 = 0.5 * accu(0.5 * log(S % S)) + n * q * 0.5;

    
   
    
    double objective = elbo1 + elbo2 + elbo3 + elbo4 + elbo5 ;
    
                        
    arma::mat gradB = GradB(vecY, X, vecR, vecxi, vecA);
    arma::mat gradD = X.t() * (vecxi - vecpi);
    //arma::mat gradC = (R % xi % (Y - A)).t() * M - (R % xi % A).t() * S % C;
    arma::mat gradC = GradC(R, xi, Y, A, M, S, C);
    //arma::mat gradM = (R % xi % (Y - A) * C - M);
    arma::mat gradM = GradM(R, xi, Y, A, M, C);
    //arma::mat gradS = 0.5 * (1. / S - 1. - R % xi % A * (C % C));
    arma::mat gradS = GradS(R, xi, A, S, C);
    //arma::mat gradS = arma::zeros<arma::mat>(S.n_rows, S.n_cols);

    
    return std::make_tuple(
        xi, elbo1, elbo2, elbo3, elbo4, elbo5, objective,
        gradB, gradD, gradC, gradM, gradS, A
    );
}


//---------------------------------------------------------------------------------------
// Version avec logS

inline std::tuple<
    arma::mat, double, double, double, double, double, double,
    arma::mat, arma::mat, arma::mat, arma::mat, arma::mat, arma::mat
>
Elbo_grad_LogS(const arma::mat & Y, const arma::mat & X, const arma::mat & R,
              const arma::mat & B, const arma::mat & D, const arma::mat & C, 
              const arma::mat & M, const arma::mat & logS, double tolxi
                ) {

    int n = Y.n_rows;
    int p = Y.n_cols;
    int q = M.n_cols;
    arma::mat S = exp(logS) ;
    auto [xi, elbo1, elbo2, elbo3, elbo4, elbo5, objective, gradB, gradD, gradC, gradM, gradS, A] = 
            Elbo_grad(Y, X, R, B, D, C, M, S, tolxi);


    gradS = S % gradS;

     return std::make_tuple(
        xi, elbo1, elbo2, elbo3, elbo4, elbo5, objective,
        gradB, gradD, gradC, gradM, gradS, A
    );
}


//------------------------------------------------------------------------------------------
// Par lignes

inline std::tuple<
    arma::mat, double, double, double, double, double, double,
    arma::vec, arma::vec, arma::mat
>
Elbo_grad_i(const arma::mat & Y, const arma::mat & X, const arma::mat & R,
            const arma::mat & B, const arma::mat & D, const arma::mat & C, 
            const arma::mat & M, const arma::mat & S, double tolxi, int i) {
    
    int n = Y.n_rows;
    int p = Y.n_cols;
    int q = M.n_cols;

    arma::vec XB = X * B;
    arma::vec XD = X * D;

    arma::mat mu = arma::reshape(XB, n, p);
    arma::mat nu = arma::reshape(XD, n, p);

    arma::mat Z = mu + M * C.t();

    arma::mat A = arma::exp(Z + 0.5 * S * arma::pow(C, 2).t());

    arma::mat log_fact_Y = log_factorial_matrix(Y);

    arma::mat pi = 1.0 / (1.0 + arma::exp(-nu));

    arma::mat xi = ifelse_mat(Y, A, nu, R, tolxi);

    // Correction ici : évaluer d'abord dans tmp
    arma::mat tmp_elbo1 = xi % nu - ifelse_exp(nu);
    double elbo1 = arma::accu(tmp_elbo1.row(i));

    arma::mat tmp_elbo2 = (M % M) + S;
    double elbo2 = -0.5 * arma::accu(tmp_elbo2.row(i));

    arma::mat etape = R % xi % (Y % (mu + M * C.t()) - A - log_fact_Y);
    etape.elem(arma::find((R == 0.0) || (xi == 0.0))).zeros();
    double elbo3 = arma::accu(etape.row(i));

    arma::mat mask = arma::conv_to<arma::mat>::from((xi > 0.0) % (xi < 1.0));
    arma::mat valid_xi = xi % mask;

    arma::mat H = -(valid_xi % arma::log(valid_xi) + (1 - valid_xi) % arma::log(1 - valid_xi));
    H.replace(arma::datum::nan, 0);
    double elbo4 = arma::accu(H.row(i));

    arma::mat tmp_elbo5 = 0.5 * arma::log(S % S);
    double elbo5 = 0.5 * arma::accu(tmp_elbo5.row(i)) + q * 0.5;

    double objective = elbo1 + elbo2 + elbo3 + elbo4 + elbo5;

    arma::vec gradM = GradM(R, xi, Y, A, M, C).row(i).t();
    arma::vec gradS = GradS(R, xi, A, S, C).row(i).t();

    return std::make_tuple(
        xi, elbo1, elbo2, elbo3, elbo4, elbo5, objective,
        gradM, gradS, A
    );
}





