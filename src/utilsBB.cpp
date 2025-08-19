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
// arma::mat ifelse_mat(const arma::mat& Y, const arma::mat& A, const arma::mat& nu, const arma::mat& R, double tolxi) {
//     arma::mat xi = arma::ones(size(Y)); // Étape 1 : Initialisation à 1 partout
// 
//     // Étape 2 : Si R == 0, alors xi = nu
//     arma::uvec mask_R0 = arma::find(R == 0.0);
//     xi.elem(mask_R0) = nu.elem(mask_R0);
// 
//     // Étape 3 : Si Y == 0 et R == 1, alors xi = nu - A * R
//     arma::uvec mask_Y0_R1 = arma::find((Y == 0) % (R == 1));
//     xi.elem(mask_Y0_R1) = nu.elem(mask_Y0_R1) - A.elem(mask_Y0_R1) % R.elem(mask_Y0_R1);
// 
//     // Étape 4 : Appliquer la transformation logistique selon le signe de xi
//     arma::uvec pos_mask = arma::find(xi >= 0);
//     arma::uvec neg_mask = arma::find(xi < 0);
// 
//     xi.elem(pos_mask) = 1 / (1 + arma::exp(-xi.elem(pos_mask)));
//     xi.elem(neg_mask) = arma::exp(xi.elem(neg_mask)) / (1 + arma::exp(xi.elem(neg_mask)));
// 
//     // Étape 5 : Appliquer la tolérance pour éviter exactement 0 et 1
//     //xi = arma::clamp(xi, tolxi, 1.0 - tolxi);
// 
//     return xi;
// }

// Définition de la fonction ifelse_mat
arma::mat ifelse_mat(const arma::mat& Y, const arma::mat& A, const arma::mat& nu, const arma::mat& R, double tolxi) {
  arma::mat xi = arma::ones(size(Y));
  
  // Masque pour Y == 0 et R == 1, ou R == 0
  arma::uvec mask = arma::find((Y == 0) % (R == 1) || (R == 0));
  
  // Calculer xi pour les éléments du masque
  xi.elem(mask) = nu.elem(mask) - A.elem(mask) % R.elem(mask);
  
  // Masque pour R == 0
  
  arma::uvec mask_R0 = arma::find(R == 0);
  xi.elem(mask_R0) = nu.elem(mask_R0);
  
  // Appliquer la transformation logistique
  arma::uvec pos_mask = find(xi.elem(mask) >= 0);
  arma::uvec neg_mask = find(xi.elem(mask) < 0);
  
  xi.elem(mask(pos_mask)) = 1 / (1 + exp(-xi.elem(mask(pos_mask))));
  xi.elem(mask(neg_mask)) = exp(xi.elem(mask(neg_mask))) / (exp(xi.elem(mask(neg_mask))) + 1);
  
  // Les autres éléments restent à 1 (valeur d'initialisation)
  
  xi.elem(mask) = arma::clamp(xi.elem(mask), tolxi, 1.0 - tolxi);
  
  return xi;
}

// Définition de la fonction ifelse_exp
arma::mat ifelse_exp(const arma::mat& nu) {
  arma::mat F(size(nu));

  // Calculer log(1 + exp(x)) de manière stable
  F = arma::log1p(arma::exp(nu));

  // Pour x > 0, utiliser x + log(1 + exp(-x)) pour éviter le débordement
  arma::uvec positive_indices = arma::find(nu > 0);
  F.elem(positive_indices) = nu.elem(positive_indices) +
    arma::log1p(arma::exp(-nu.elem(positive_indices)));

  return F;
}


// Définition de la fonction entropie_logis
double entropie_logis(const arma::mat& xi) {
  // Créer un masque pour les valeurs valides (entre 0 et 1, exclusivement)
  arma::mat mask = arma::conv_to<arma::mat>::from((xi > 0.0) % (xi < 1));

  // Appliquer le masque à xi
  arma::mat valid_xi = xi % mask;

  // Calculer l'entropie pour les éléments valides
  arma::mat H = -(valid_xi % arma::log(valid_xi) + (1 - valid_xi) % arma::log(1 - valid_xi));

  // Remplacer les valeurs NaN par 0
  H.replace(arma::datum::nan, 0);

  // Sommer toutes les contributions à l'entropie
  return arma::accu(H);
}


// Définition de la fonction GradB
arma::mat GradB(const arma::vec & vecY, const arma::mat & X, const arma::vec & vecR, const arma::vec & vecxi, const arma::vec & vecA) {
  int l = vecY.size();

  arma::vec gradi = vecR % vecxi % (vecY - vecA);

  // Créer un masque pour les indices où vecR et vecxi sont non nuls
  arma::uvec mask = (vecR != 0.0) && (vecxi != 0.0);

  // Mettre à zéro les gradients là où le masque est faux
  gradi.elem(find(mask == 0)).zeros();

  arma::mat gradB = X.t() * gradi;
  return gradB;
}


// Définition de la fonction GradC
arma::mat GradC(const arma::mat & R, const arma::mat & xi, const arma::mat & Y, const arma::mat & A,
                const arma::mat & M, const arma::mat & S, const arma::mat & C) {
  int n = Y.n_rows;
  int p = Y.n_cols;
  // Calcul des gradients
  arma::mat grad1 = R % xi % (Y - A);
  arma::mat grad2 = R % xi % A;

  // Créer un masque logique pour les éléments où R ou xi sont nuls
  arma::uvec mask1 = find(R == 0.0);
  arma::uvec mask2 = find(xi == 0.0);
  arma::uvec mask = join_vert(mask1, mask2);
  mask = unique(mask);

  // Mettre à zéro les éléments de grad1 et grad2 là où le masque est vrai
  grad1.elem(mask).zeros();
  grad2.elem(mask).zeros();

  // Calcul final
  arma::mat gradC = grad1.t() * M - grad2.t() * S % C;
  return gradC;
}


// Définition de la fonction GradM
arma::mat GradM(const arma::mat & R, const arma::mat & xi, const arma::mat & Y, const arma::mat & A,
                const arma::mat & M, const arma::mat & C) {
  int n = Y.n_rows;
  int p = Y.n_cols;

  // Calcul du gradient
  arma::mat grad1 = R % xi % (Y - A);

  // Créer un masque pour les éléments où R et xi sont non nuls
  arma::uvec mask1 = find(R != 0.0);
  arma::uvec mask2 = find(xi != 0.0);
  arma::uvec mask = join_vert(mask1, mask2);
  mask = unique(mask);

  // Mettre à zéro les éléments de grad1 là où le masque est faux
  grad1.elem(find(mask == 0)).zeros();

  // Calcul final
  arma::mat gradM = grad1 * C - M;

  return gradM;
}


// Définition de la fonction GradS
arma::mat GradS(const arma::mat & R, const arma::mat & xi, const arma::mat & A,
                const arma::mat & S, const arma::mat & C) {
  int n = A.n_rows;
  int p = A.n_cols;

  // Calcul du gradient
  arma::mat grad1 = R % xi % A;

  // Créer un masque pour les éléments où R et xi sont non nuls
  arma::uvec mask1 = find(R != 0.0);
  arma::uvec mask2 = find(xi != 0.0);
  arma::uvec mask = join_vert(mask1, mask2);
  mask = unique(mask);

  // Mettre à zéro les éléments de grad1 là où le masque est faux
  grad1.elem(find(mask == 0)).zeros();

  // Calcul final
  arma::mat gradS = 0.5 * (1. / S - 1. - grad1 * (C % C));

  return gradS;
}


// Définition de la fonction Elbo3

double Elbo3(const arma::mat & R, const arma::mat & xi, const arma::mat & mu,
             const arma::mat & Y, const arma::mat & M, const arma::mat & C,
             const arma::mat & A, const arma::mat & log_fact_Y) {

  // Calcul de l'ELBO
  arma::mat elbo = R % xi % (Y % (mu + M * C.t()) - A - log_fact_Y);

  // Mettre à zéro les éléments de elbo là où R ou xi sont nuls
  elbo.elem(arma::find(R == 0.0 || xi == 0.0)).zeros();

  // Somme des éléments pour obtenir elbo3
  double elbo3 = accu(elbo);

  return elbo3;
}

double Elbo3_q0(const arma::mat & R, const arma::mat & xi, const arma::mat & mu,
             const arma::mat & Y,
             const arma::mat & A, const arma::mat & log_fact_Y) {

  // Calcul de l'ELBO
  arma::mat elbo = R % xi % (Y % mu - A - log_fact_Y);

  // Mettre à zéro les éléments de elbo là où R ou xi sont nuls
  elbo.elem(arma::find(R == 0.0 || xi == 0.0)).zeros();

  // Somme des éléments pour obtenir elbo3
  double elbo3 = accu(elbo);

  return elbo3;
}





