// Copyright 2023 Brian Howell
// MIT License
// Project: BayesGA

#ifndef SRC_GENETICALGORITHM_H_
#define SRC_GENETICALGORITHM_H_

#include "Optimizable.h"
#include "common.h"

class GeneticAlgorithm {
public:
    GeneticAlgorithm(Optimizable &sim, 
                     bopt        &init_bopt, 
                     constraints &ga_constraints);

    void optimize();

private:
    Optimizable _sim;
    bopt        _init_bopt;
    constraints _ga_constraints;

};

#endif // SRC_GENETICALGORITHM_H_