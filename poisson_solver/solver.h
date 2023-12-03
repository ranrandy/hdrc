#ifndef POISSON_SOLVER_H_INCLUDED
#define POISSON_SOLVER_H_INCLUDED

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
    Jacobi iterative method:
        x_i^{k+1} = \frac{1}{-4} (divG_i - x_{i-1}^k - x_{i-W}^k - x_{i+1}^k - x_{i+W}^k).
*/
void jacobiSolver(
    const int H, const int W, 
    const float* d_divG,
    const int iterations, const float tolerance, const int checkFrequency,
    float* d_I_log
);

/*
    Gauss-Seidel iterative method:
        x_i^{k+1} = \frac{1}{-4} (divG_i - x_{i-1}^{k+1} - x_{i-W}^{k+1} - x_{i+1}^k - x_{i+W}^k).
*/
void gaussSeidelSolver(
    const int H, const int W, 
    const float* d_divG,
    const int iterations, const float tolerance, const int checkFrequency,
    float* d_I_log
);

/*
    Gauss-Seidel iterative method with Red-Black Reordering:
        x_{2i}^{k+1} = \frac{1}{-4} (divG_i - x_{i-1}^k - x_{i-W}^k - x_{i+1}^k - x_{i+W}^k),
        x_{2i+1}^{k+1} = \frac{1}{-4} (divG_i - x_{i-1}^{k+1} - x_{i-W}^{k+1} - x_{i+1}^{k+1} - x_{i+W}^{k+1}).
*/
void gaussSeidelRedBlackSolver(
    const int H, const int W, 
    const float* d_divG,
    const int iterations, const float tolerance, const int checkFrequency,
    float* d_I_log
);

/*
    Gauss-Seidel iterative method with Red-Black Reordering and Overrelaxation (SOR):
        x_{2i}^{k+1} = \frac{1}{-4} (divG_i - w_{opt} * x_{i-1}^k - w_{opt} * x_{i-W}^k - x_{i+1}^k - x_{i+W}^k),
        x_{2i+1}^{k+1} = \frac{1}{-4} (divG_i - w_{opt} * x_{i-1}^{k+1} - w_{opt} * x_{i-W}^{k+1} - x_{i+1}^{k+1} - x_{i+W}^{k+1}),
        w_{opt} = \frac{2}{1 + \sqrt{1 - \cos{\pi/\frac{H+W}{2}}^2}}.
*/
void gaussSeidelRedBlackSORSolver(
    const int H, const int W, 
    const float* d_divG,
    const int iterations, const float tolerance, const int checkFrequency,
    float* d_I_log
);

/*
    Full Multigrid Poisson Solver with Gauss-Seidel smoothing iteration + Red-Black Reordering and Overrelaxation (SOR).
*/
void fullMultigridSolver(
    const int H, const int W, 
    const float* d_divG, 
    const int iterations, const float tolerance, const int checkFrequency,
    float* d_I_log);

#endif