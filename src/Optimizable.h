// Copyright 2023 Brian Howell
// MIT License
// Project: BayesGA

#ifndef SRC_OPTIMIZABLE_H_
#define SRC_OPTIMIZABLE_H_

class Optimizable {
public:
    virtual void simulate() = 0;
    virtual const double& getObjective() const = 0;
};

#endif //UGAPDIFFUSION_VOXEL_H
