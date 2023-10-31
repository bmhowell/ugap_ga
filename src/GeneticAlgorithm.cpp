// Copyright 2023 Brian Howell
// MIT License
// Project: BayesGA

#include "GeneticAlgorithm.h"

GeneticAlgorithm::GeneticAlgorithm(Optimizable& sim, 
                                   bopt        &init_bopt, 
                                   constraints &ga_constraints
                                   ) : 
                  _sim(sim), _init_bopt(init_bopt), _ga_constraints(ga_constraints) {}

void GeneticAlgorithm::optimize() {

    void optimize(Simulation* simulation) {
        // Implement your genetic algorithm for optimization using the simulation object.
        // You can call simulation->simulate() to run the simulation and simulation->get_objective() to get the objective value.
        // Use constraints_ to generate sample input parameters.
    }
}