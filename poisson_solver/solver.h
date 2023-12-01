#ifndef POISSON_SOLVER_H_INCLUDED
#define POISSON_SOLVER_H_INCLUDED

#include <cuda_runtime.h>
// #include <iostream>

void jacobiSolver(
    const int H, const int W, 
    const float* divG,
    const int iterations, const float tolerence,
    float* I_log
);

void fullMultigridSolver(
    const int H, const int W, 
    const float* divG, 
    float* I_log);

#endif