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


// Sortie R de l'Elbo et des gradients


// [[Rcpp::export]]
Rcpp::List Elbo_grad_logS_Rcpp(const Rcpp::List & data, // List(Y, R, X)
                 const Rcpp::List & params // List(B, C, M, logS)
                ) {
    const arma::mat & Y = Rcpp::as<arma::mat>(data["Y"]); // responses (n,p)
    const arma::mat & R = Rcpp::as<arma::mat>(data["R"]); // missing data (n,p)
    const arma::mat & X = Rcpp::as<arma::mat>(data["X"]); // covariates (np,d)
    const arma::mat & B = Rcpp::as<arma::mat>(params["B"]); // (1,d) régresseurs pour la Poisson
    const arma::mat & D = Rcpp::as<arma::mat>(params["D"]); // (1,d) régresseurs pour la logistique
    const arma::mat & C = Rcpp::as<arma::mat>(params["C"]); // (p,q)
    const arma::mat & M = Rcpp::as<arma::mat>(params["M"]); // (n,q)
    const arma::mat & logS = Rcpp::as<arma::mat>(params["logS"]); // (n,q)




    auto [xi, elbo1, elbo2, elbo3, elbo4, elbo5, objective, gradB, gradD, gradC, gradM, gradS, A] = 
            Elbo_grad_LogS(Y, X, R, B, D, C, M, logS);


    return Rcpp::List::create(
    	Rcpp::Named("xi", xi),
        Rcpp::Named("elbo1", elbo1),
        Rcpp::Named("elbo2", elbo2),
        Rcpp::Named("elbo3", elbo3),
        Rcpp::Named("elbo4", elbo4),
        Rcpp::Named("elbo5", elbo5),
        Rcpp::Named("objective", objective),
        Rcpp::Named("gradB", gradB),
        Rcpp::Named("gradD", gradD),
        Rcpp::Named("gradC", gradC),
        Rcpp::Named("gradM", gradM),
        Rcpp::Named("gradS", gradS),
        Rcpp::Named("A", A)
    );
}


//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List nlopt_optimize_ZIP_logS(
    const Rcpp::List & data  , // List(Y, R, X)
    const Rcpp::List & params, // List(B, C, M, S)
    const Rcpp::List & config // List of config values
) {
    // Conversion from R, prepare optimization
    const arma::mat & Y = Rcpp::as<arma::mat>(data["Y"]); // responses (n,p)
    const arma::mat & R = Rcpp::as<arma::mat>(data["R"]); // missing data (n,p)
    const arma::mat & X = Rcpp::as<arma::mat>(data["X"]); // covariates (np,d)
    const auto init_B = Rcpp::as<arma::mat>(params["B"]); // (1,d) régresseurs pour la Poisson
    const auto init_D = Rcpp::as<arma::mat>(params["D"]); // (1,d) régresseurs pour la logistique
    const auto init_C = Rcpp::as<arma::mat>(params["C"]); // (p,q)
    const auto init_M = Rcpp::as<arma::mat>(params["M"]); // (n,q)
    const auto init_logS = Rcpp::as<arma::mat>(params["logS"]); // (n,q)




    const auto metadata = tuple_metadata(init_B, init_D, init_C, init_M, init_logS);
    enum { B_ID, D_ID, C_ID, M_ID, logS_ID }; // Names for metadata indexes

    auto parameters = std::vector<double>(metadata.packed_size);
    metadata.map<B_ID>(parameters.data()) = init_B;
    metadata.map<D_ID>(parameters.data()) = init_D;
    metadata.map<C_ID>(parameters.data()) = init_C;
    metadata.map<M_ID>(parameters.data()) = init_M;
    metadata.map<logS_ID>(parameters.data()) = init_logS;

    auto optimizer = new_nlopt_optimizer(config, parameters.size());


    if(config.containsElementNamed("xtol_abs")) {
        SEXP value = config["xtol_abs"];
        if(Rcpp::is<double>(value)) {
            set_uniform_xtol_abs(optimizer.get(), Rcpp::as<double>(value));
        } else {
            auto per_param_list = Rcpp::as<Rcpp::List>(value);
            auto packed = std::vector<double>(metadata.packed_size);
            set_from_r_sexp(metadata.map<B_ID>(packed.data()), per_param_list["B"]);
            set_from_r_sexp(metadata.map<D_ID>(packed.data()), per_param_list["D"]);
            set_from_r_sexp(metadata.map<C_ID>(packed.data()), per_param_list["C"]);
            set_from_r_sexp(metadata.map<M_ID>(packed.data()), per_param_list["M"]);
            set_from_r_sexp(metadata.map<logS_ID>(packed.data()), per_param_list["logS"]);
            set_per_value_xtol_abs(optimizer.get(), packed);
        }
    }

    std::vector<double> objective_values;



 // Optimize
    auto objective_and_grad = [&metadata, &X, &Y, &R, &objective_values](const double * params, double * grad) -> double {
        const arma::mat B = metadata.map<B_ID>(params);
        const arma::mat D = metadata.map<D_ID>(params);
        const arma::mat C = metadata.map<C_ID>(params);
        const arma::mat M = metadata.map<M_ID>(params);
        const arma::mat logS = metadata.map<logS_ID>(params);
        
        
    auto [xi, elbo1, elbo2, elbo3, elbo4, elbo5, objective, gradB, gradD, gradC, gradM, gradS, A] = 
    Elbo_grad_LogS(Y, X, R, B, D, C, M, logS);
    
    objective = -objective;

        objective_values.push_back(- objective);
        
        arma::vec vecout = {elbo1, elbo2, elbo3, elbo4, elbo5, objective};
        //std::cout << objective << std::endl;

        metadata.map<B_ID>(grad) = - gradB;
        metadata.map<D_ID>(grad) = - gradD;
	metadata.map<C_ID>(grad) = - gradC;
        metadata.map<M_ID>(grad) = - gradM;
        metadata.map<logS_ID>(grad) =  - gradS;
        

        return objective;
    };
    OptimizerResult result = minimize_objective_on_parameters(optimizer.get(), objective_and_grad, parameters);

    // Model and variational parameters
    arma::mat B = metadata.copy<B_ID>(parameters.data());
    arma::mat D = metadata.copy<D_ID>(parameters.data());
    arma::mat C = metadata.copy<C_ID>(parameters.data());
    arma::mat M = metadata.copy<M_ID>(parameters.data());
    arma::mat logS = metadata.copy<logS_ID>(parameters.data());
    
        auto [xi, elbo1, elbo2, elbo3, elbo4, elbo5, objective, gradB, gradD, gradC, gradM, gradS, A] = 
    Elbo_grad_LogS(Y, X, R, B, D, C, M, logS);
    
  	    

    return Rcpp::List::create(
        Rcpp::Named("B", B),
        Rcpp::Named("D", D),
        Rcpp::Named("C", C),
        Rcpp::Named("M", M),
        Rcpp::Named("S", exp(logS)),
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
