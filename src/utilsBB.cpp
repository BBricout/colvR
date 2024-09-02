#include "utilsBB.h"
#include <Rcpp.h>

// Définition de la fonction MatrixToVector
// [[Rcpp::export]]
Rcpp::NumericVector MatrixToVector(const arma::mat & matrix) {
    int n = matrix.n_rows;
    int p = matrix.n_cols;

    if (n == 0 || p == 0) {
        Rcpp::stop("Input matrix is empty");
    }

    arma::vec vectorized = arma::vectorise(matrix);
    Rcpp::NumericVector result(vectorized.begin(), vectorized.end());

    return result;
}

// Définition de la fonction VectorToMatrix
// [[Rcpp::export]]
Rcpp::NumericMatrix VectorToMatrix(const arma::vec & vector, int n, int p) {
    arma::mat matrix = arma::reshape(vector, n, p);
    Rcpp::NumericMatrix result(n, p, matrix.memptr()); 
  
    return result;
}

// Définition de la fonction log_factorial
double log_factorial(double n) {
    return lgamma(n + 1);
}

// Définition de la fonction log_factorial_matrix
arma::mat log_factorial_matrix(const arma::mat& Y) {
    arma::mat result = Y;
    result.transform([](double val) { return log_factorial(val); });
    return result;
}

// Définition de la fonction ifelse_mat
arma::mat ifelse_mat(const arma::mat& Y, const arma::mat& A, const arma::mat& nu, const arma::mat& R) {
    int n = Y.n_rows;
    int p = Y.n_cols;
    
    arma::Mat<double> xi(n, p, arma::fill::none); 
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j) {
            if (Y(i,j)*R(i, j) == 0){
            	if (R(i,j)==0){
           		xi(i,j) = nu(i,j);
            	}
            	else{
            		xi(i,j) = nu(i,j) - A(i,j);
            	}
                if (xi(i,j) >= 0){
                	xi(i, j) = 1./(1. + exp(-xi(i, j)));
                } else {
                	xi(i,j) = exp(xi(i,j))/ (exp(xi(i,j)) + 1.);
                }
            } else {
                xi(i,j) = 1.;
            }
        }
    }
    
   return xi;
}

// Définition de la fonction ifelse_exp
arma::mat ifelse_exp(const arma::mat& nu) {
    int n = nu.n_rows;
    int p = nu.n_cols;
    
    arma::Mat<double> F(n, p, arma::fill::none); 
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j) {
            if (nu(i,j) <= 0) {
                F(i,j) = log(exp(nu(i,j)) + 1.);
            } else {
                F(i,j) = log((1. + exp(-nu(i,j)))/exp(-nu(i,j)));
            }
        }
    }
   
   return F;
}

// Définition de la fonction entropie_logis
double entropie_logis(arma::mat & xi) {
    int n = xi.n_rows;
    int p = xi.n_cols;
    
    double H = 0.;
    
    for (size_t i = 0; i < n; ++i) {
    	for (size_t j = 0; j < p; ++j){
    		if (xi(i,j) != 0. && xi(i,j) != 1.){
    			H = H - (xi(i,j) * log(xi(i,j)) + (1 - xi(i,j)) * log(1 - xi(i,j))); 
    		}
    	}
    }
    return H;
}

// Définition de la fonction GradB
arma::mat GradB(const arma::vec & vecY, const arma::mat & X, const arma::vec & vecR, const arma::vec & vecxi, const arma::vec & vecA) {
	int l = vecY.size();
	
	arma::vec gradi = vecR % vecxi % (vecY - vecA);
	for (size_t i = 0; i < l; ++i) {
		if (vecR[i] * vecxi[i] == 0) {
			gradi[i] = 0;
		}
	}
	arma::mat gradB = X.t() * gradi;
	
	return gradB;
}

// Définition de la fonction GradC
arma::mat GradC(const arma::mat & R, const arma::mat & xi, const arma::mat & Y, const arma::mat & A,
		const arma::mat & M, const arma::mat & S, const arma::mat & C) {
	int n = Y.n_rows;
	int p = Y.n_cols;
		
	arma::mat grad1 = R % xi % (Y - A);
	arma::mat grad2 = R % xi % A;
		
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < p; ++j) {
			if (R(i,j) * xi(i,j) == 0) {
				grad1(i,j) = 0;
				grad2(i,j) = 0;
			}
		}
	}
		
	arma::mat gradC = grad1.t() * M - grad2.t() * S % C;
		
	return gradC;
}

// Définition de la fonction GradM
arma::mat GradM(const arma::mat & R, const arma::mat & xi, const arma::mat & Y, const arma::mat & A,
		const arma::mat & M, const arma::mat & C) {
	int n = Y.n_rows;
	int p = Y.n_cols;
		
	arma::mat grad1 = R % xi % (Y - A);
		
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < p; ++j) {
			if (R(i,j) * xi(i,j) == 0) {
				grad1(i,j) = 0;
			}
		}
	}
		
	arma::mat gradM = grad1 * C - M;
		
	return gradM;
}

// Définition de la fonction GradS
arma::mat GradS(const arma::mat & R, const arma::mat & xi, const arma::mat & A,
		const arma::mat & S, const arma::mat & C) {
	int n = A.n_rows;
	int p = A.n_cols;
		
	arma::mat grad1 = R % xi % A;
		
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < p; ++j) {
			if (R(i,j) * xi(i,j) == 0) {
				grad1(i,j) = 0;
			}
		}
	}
		
	arma::mat gradS = 0.5 * (1. / S - 1. - grad1 * (C % C));
		
	return gradS;
}

// Définition de la fonction Elbo3
double Elbo3(const arma::mat & R, const arma::mat & xi, const arma::mat & mu,
		const arma::mat & Y, const arma::mat & M, const arma::mat & C,
		const arma::mat & A, const arma::mat & log_fact_Y) {
	int n = A.n_rows;
	int p = A.n_cols;
		
	arma::mat elbo = R % xi % (Y % (mu + M * C.t()) - A - log_fact_Y);
		
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < p; ++j) {
			if (R(i,j) * xi(i,j) == 0) {
				elbo(i,j) = 0;
			}
		}
	}
		
	double elbo3 = accu(elbo);
		
	return elbo3;
}

