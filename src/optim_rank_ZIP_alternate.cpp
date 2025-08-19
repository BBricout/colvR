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

using namespace arma;

// Structure pour gérer les blocs actifs
struct ActiveBlocks {
    bool B = false;
    bool D = false;
    bool C = false;
    bool M = false;
    bool S = false;
};

// [[Rcpp::export]]
Rcpp::List nlopt_optimize_ZIP_Steps(
    const Rcpp::List & data,    // List(Y, R, X)
    const Rcpp::List & params,  // List(B, C, M, S, D)
    const Rcpp::List & config,  // List contenant xtol_abs, bounds, etc.
    double tolxi,
    Rcpp::List active_block_list // List(B=, D=, C=, M=, S=) pour activer les blocs
) {
    // Données
    const mat & Y = Rcpp::as<mat>(data["Y"]);
    const mat & R = Rcpp::as<mat>(data["R"]);
    const mat & X = Rcpp::as<mat>(data["X"]);

    // Paramètres
    const auto init_B = Rcpp::as<mat>(params["B"]);
    const auto init_D = Rcpp::as<mat>(params["D"]);
    const auto init_C = Rcpp::as<mat>(params["C"]);
    const auto init_M = Rcpp::as<mat>(params["M"]);
    const auto init_S = Rcpp::as<mat>(params["S"]);

    // Initialiser metadata
    const auto metadata = tuple_metadata(init_B, init_D, init_C, init_M, init_S);
    enum { B_ID, D_ID, C_ID, M_ID, S_ID };

    // Vectoriser les paramètres
    std::vector<double> parameters(metadata.packed_size);
    metadata.map<B_ID>(parameters.data()) = init_B;
    metadata.map<D_ID>(parameters.data()) = init_D;
    metadata.map<C_ID>(parameters.data()) = init_C;
    metadata.map<M_ID>(parameters.data()) = init_M;
    metadata.map<S_ID>(parameters.data()) = init_S;

    // Optimiseur
    auto optimizer = new_nlopt_optimizer(config, parameters.size());

    // Bornes
    std::vector<double> lower_bounds(metadata.packed_size, -HUGE_VAL);
    std::vector<double> upper_bounds(metadata.packed_size, HUGE_VAL);

    if (config.containsElementNamed("lower_bounds")) {
        lower_bounds = Rcpp::as<std::vector<double>>(config["lower_bounds"]);
    }
    if (config.containsElementNamed("upper_bounds")) {
        upper_bounds = Rcpp::as<std::vector<double>>(config["upper_bounds"]);
    }

    nlopt_set_lower_bounds(optimizer.get(), lower_bounds.data());
    nlopt_set_upper_bounds(optimizer.get(), upper_bounds.data());

    // Tolérance
    if (config.containsElementNamed("xtol_abs")) {
        SEXP value = config["xtol_abs"];
        if (Rcpp::is<double>(value)) {
            set_uniform_xtol_abs(optimizer.get(), Rcpp::as<double>(value));
        } else {
            auto per_block = Rcpp::as<Rcpp::List>(value);
            std::vector<double> packed(metadata.packed_size);
            set_from_r_sexp(metadata.map<B_ID>(packed.data()), per_block["B"]);
            set_from_r_sexp(metadata.map<D_ID>(packed.data()), per_block["D"]);
            set_from_r_sexp(metadata.map<C_ID>(packed.data()), per_block["C"]);
            set_from_r_sexp(metadata.map<M_ID>(packed.data()), per_block["M"]);
            set_from_r_sexp(metadata.map<S_ID>(packed.data()), per_block["S"]);
            set_per_value_xtol_abs(optimizer.get(), packed);
        }
    }

    // Bloc actif
    ActiveBlocks active;
    active.B = Rcpp::as<bool>(active_block_list["B"]);
    active.D = Rcpp::as<bool>(active_block_list["D"]);
    active.C = Rcpp::as<bool>(active_block_list["C"]);
    active.M = Rcpp::as<bool>(active_block_list["M"]);
    active.S = Rcpp::as<bool>(active_block_list["S"]);

    std::vector<double> objective_values;

    // Fonction objectif + gradient
    auto objective_and_grad = [&](const double *params, double *grad) -> double {
        const mat B = metadata.map<B_ID>(params);
        const mat D = metadata.map<D_ID>(params);
        const mat C = metadata.map<C_ID>(params);
        const mat M = metadata.map<M_ID>(params);
        const mat S = metadata.map<S_ID>(params);

        auto [xi, elbo1, elbo2, elbo3, elbo4, elbo5, obj, gradB, gradD, gradC, gradM, gradS, A] =
            Elbo_grad(Y, X, R, B, D, C, M, S, tolxi);
            
            int n = Y.n_rows;
	    int p = Y.n_cols;
	    int q = M.n_cols;
	    int d = X.n_cols;

        obj = -obj;
        objective_values.push_back(-obj);

        if (grad) {
	    	metadata.map<B_ID>(grad) = active.B ? (-gradB).eval() : arma::zeros<arma::mat>(d, 1);
		metadata.map<D_ID>(grad) = active.D ? (-gradD).eval() : arma::zeros<arma::mat>(d, 1);
		metadata.map<C_ID>(grad) = active.C ? (-gradC).eval() : arma::zeros<arma::mat>(p, q);
		metadata.map<M_ID>(grad) = active.M ? (-gradM).eval() : arma::zeros<arma::mat>(n, q);
		metadata.map<S_ID>(grad) = active.S ? (-gradS).eval() : arma::zeros<arma::mat>(n, q);

            }
            

        return obj;
    };

    // Lancement optimisation
    OptimizerResult result = minimize_objective_on_parameters(optimizer.get(), objective_and_grad, parameters);

    // Reconstruction des paramètres optimisés
    mat B = metadata.copy<B_ID>(parameters.data());
    mat D = metadata.copy<D_ID>(parameters.data());
    mat C = metadata.copy<C_ID>(parameters.data());
    mat M = metadata.copy<M_ID>(parameters.data());
    mat S = metadata.copy<S_ID>(parameters.data());

    auto [xi, elbo1, elbo2, elbo3, elbo4, elbo5, final_obj, _, __, ___, ____, _____, A] =
        Elbo_grad(Y, X, R, B, D, C, M, S, tolxi);

    return Rcpp::List::create(
        Rcpp::Named("B") = B,
        Rcpp::Named("D") = D,
        Rcpp::Named("C") = C,
        Rcpp::Named("M") = M,
        Rcpp::Named("S") = S,
        Rcpp::Named("A") = A,
        Rcpp::Named("xi") = xi,
        Rcpp::Named("elbo1") = elbo1,
        Rcpp::Named("elbo2") = elbo2,
        Rcpp::Named("elbo3") = elbo3,
        Rcpp::Named("elbo4") = elbo4,
        Rcpp::Named("elbo5") = elbo5,
        Rcpp::Named("objective") = final_obj,
        Rcpp::Named("objective_values") = objective_values,
        Rcpp::Named("monitoring") = Rcpp::List::create(
            Rcpp::Named("status") = static_cast<int>(result.status),
            Rcpp::Named("backend") = "nlopt",
            Rcpp::Named("iterations") = result.nb_iterations
        )
    );
}

