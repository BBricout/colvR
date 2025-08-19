#include "RcppArmadillo.h"
#include <cmath>
#include <iostream>
#include <Rcpp.h>
#include <nlopt.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(nloptr)]]
// [[Rcpp::plugins(cpp11)]]

#include "nlopt_wrapper.h"
#include "packing.h"
#include "utils.h"
#include "utilsBB.h"
#include "Elbo_gradBB.h"

//--------------------------------------------------------------------------------------------------------------------
// Optimisation

// On donne en entrée Mi et Si

// [[Rcpp::export]]
Rcpp::List nlopt_optimize_ZIP_VE_Mi(
    const Rcpp::List & data  , // List(Y, R, X)
    const Rcpp::List & params, // List(B, D, C, Mi, Si)
    const Rcpp::List & config,  // List of config values
    double tolxi, // Tolérance sur xi
    int i // Numéro de la ligne à optimiser
) {
    // Conversion from R, prepare optimization
    const arma::mat & Y = Rcpp::as<arma::mat>(data["Y"]); // responses (n,p)
    const arma::mat & R = Rcpp::as<arma::mat>(data["R"]); // missing data (n,p)
    const arma::mat & X = Rcpp::as<arma::mat>(data["X"]); // covariates (np,d)
    const auto B = Rcpp::as<arma::mat>(params["B"]); // (1,d) régresseurs pour la Poisson
    const auto D = Rcpp::as<arma::mat>(params["D"]); // (1,d) régresseurs pour la logistique
    const auto C = Rcpp::as<arma::mat>(params["C"]); // (p,q)
    arma::mat M = Rcpp::as<arma::mat>(params["M"]); // (n,q)
    arma::mat S = Rcpp::as<arma::mat>(params["S"]); // (n,q)
    
    arma::vec init_Mi = M.row(i).t(); 
    arma::vec init_Si = S.row(i).t(); 

    const auto metadata = tuple_metadata(init_Mi, init_Si);
    enum { Mi_ID, Si_ID }; // Names for metadata indexes

    auto parameters = std::vector<double>(metadata.packed_size);
    metadata.map<Mi_ID>(parameters.data()) = init_Mi;
    metadata.map<Si_ID>(parameters.data()) = init_Si;

    auto optimizer = new_nlopt_optimizer(config, parameters.size());
    
    	  // Définition des bornes inférieures pour tous les paramètres
    if (config.containsElementNamed("lower_bounds")) {
        auto lower_bounds_r = Rcpp::as<std::vector<double>>(config["lower_bounds"]);
        if (lower_bounds_r.size() != metadata.packed_size) {
            Rcpp::stop("La taille du vecteur lower_bounds ne correspond pas à la taille totale des paramètres.");
        }
        std::vector<double> lower_bounds = lower_bounds_r;

        // Application des bornes à l'optimiseur
        nlopt_set_lower_bounds(optimizer.get(), lower_bounds.data());
    } else {
        std::vector<double> lower_bounds(metadata.packed_size, -HUGE_VAL);
        nlopt_set_lower_bounds(optimizer.get(), lower_bounds.data());
    }
    
	     // Définition des bornes supérieures pour tous les paramètres
	if (config.containsElementNamed("upper_bounds")) {
	    auto upper_bounds_r = Rcpp::as<std::vector<double>>(config["upper_bounds"]);
	    if (upper_bounds_r.size() != metadata.packed_size) {
		Rcpp::stop("La taille du vecteur upper_bounds ne correspond pas à la taille totale des paramètres.");
	    }
	    std::vector<double> upper_bounds = upper_bounds_r;

	    // Application des bornes à l'optimiseur
	    nlopt_set_upper_bounds(optimizer.get(), upper_bounds.data());
	} else {
	    std::vector<double> upper_bounds(metadata.packed_size, HUGE_VAL);
	    nlopt_set_upper_bounds(optimizer.get(), upper_bounds.data());
	}
	    
 
    if(config.containsElementNamed("xtol_abs")) {
        SEXP value = config["xtol_abs"];
        if(Rcpp::is<double>(value)) {
            set_uniform_xtol_abs(optimizer.get(), Rcpp::as<double>(value));
        } else {
            auto per_param_list = Rcpp::as<Rcpp::List>(value);
            auto packed = std::vector<double>(metadata.packed_size);
            set_from_r_sexp(metadata.map<Mi_ID>(packed.data()), per_param_list["Mi"]);
            set_from_r_sexp(metadata.map<Si_ID>(packed.data()), per_param_list["Si"]);
            set_per_value_xtol_abs(optimizer.get(), packed);
        }
    }
    
    std::vector<double> objective_values;
    
    
    
        
    

    // Optimize
    auto objective_and_grad = [&metadata, &X, &Y, &R, &B, &D, &C, &M, &S, &objective_values, &tolxi, i](const double * params, double * grad) -> double {
  
        const arma::vec Mi = metadata.map<Mi_ID>(params);
	const arma::vec Si = metadata.map<Si_ID>(params);
        
        
	// Injecter dans M et S
	arma::mat M_updated = M;
	arma::mat S_updated = S;
	M_updated.row(i) = Mi.t();
	S_updated.row(i) = Si.t();
	

    auto [xi, elbo1, elbo2, elbo3, elbo4, elbo5, objective, gradM, gradS, A] = 
    Elbo_grad_i(Y, X, R, B, D, C, M_updated, S_updated, tolxi, i);
    
    int q = M.n_cols;
   
    
    objective = -objective;
    
    

        objective_values.push_back(- objective);
       
        arma::vec vecout = {elbo1, elbo2, elbo3, elbo4, elbo5, objective};

        metadata.map<Mi_ID>(grad) = - gradM ;
        metadata.map<Si_ID>(grad) = arma::zeros(q,1);

        

        return objective;
    };
    OptimizerResult result = minimize_objective_on_parameters(optimizer.get(), objective_and_grad, parameters);

    // Model and variational parameters
    

    arma::vec Mi = metadata.copy<Mi_ID>(parameters.data());
    arma::vec Si = metadata.copy<Si_ID>(parameters.data());
    
       arma::rowvec Mi_row = Mi.t();  // Transposition explicite
	M.row(i) = Mi_row;

	arma::rowvec Si_row = Si.t();
	S.row(i) = Si_row;
      
      auto [xi, elbo1, elbo2, elbo3, elbo4, elbo5, objective, gradB, gradD, gradC, gradM, gradS, A] = 
    Elbo_grad(Y, X, R, B, D, C, M, S, tolxi);
    
  	    

    return Rcpp::List::create(
    	Rcpp::Named("elbo1", elbo1),
    	Rcpp::Named("elbo2", elbo2),
    	Rcpp::Named("elbo3", elbo3),
    	Rcpp::Named("elbo4", elbo4),
    	Rcpp::Named("elbo5", elbo5),
        Rcpp::Named("B", B),
        Rcpp::Named("D", D),
        Rcpp::Named("C", C),
        Rcpp::Named("Mi", Mi),
        Rcpp::Named("Si", Si),
        Rcpp::Named("A", A),
        Rcpp::Named("xi", xi),
        Rcpp::Named("objective", objective),
        Rcpp::Named("objective_values", objective_values),
        Rcpp::Named("monitoring", Rcpp::List::create(
            Rcpp::Named("status", static_cast<int>(result.status)),
            Rcpp::Named("backend", "nlopt"),
            Rcpp::Named("iterations", result.nb_iterations)
        ))
    );
}

