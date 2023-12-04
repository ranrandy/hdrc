#include <iostream>
#include <string>

#include "solvers.h"
#include "debug_function2D.h"

int main(int argc, char* argv[]) {
    int iterations = std::stoi(argv[1]), checkFrequency = std::stoi(argv[2]); 
    float tolerance = std::stof(argv[3]);
    std::cout << "iterations: " << iterations << ", checkFrequency:" << checkFrequency << ", tolerance:" << tolerance << std::endl;
    // --------------------------------------------- 2D ---------------------------------------------
    int W = 1600, H = 1200;
    Function2D function2d(W, H);
    
    // int j = 9, i = 25;
    // float real_laplacian = function2d._real_f_laplacian[j * W + i];
    // float estimate_laplacian = function2d._estimated_f_laplacian[j * W + i];

    // std::cout << "f(x) = sin(pi*(x+y)/100.0): " << std::endl;
    // std::cout << "Real Laplacian " << std::setprecision (15) << real_laplacian << std::endl;
    // std::cout << "Estimated Laplacian " << std::setprecision (15) << estimate_laplacian << std::endl;
    // std::cout << "L1 Error " << std::setprecision (15) << fabs(estimate_laplacian - real_laplacian) << std::endl;

    float *d_divG;
    cudaMalloc(&d_divG, H * W * sizeof(float));
    cudaMemcpy(d_divG, function2d._real_f_laplacian, H * W * sizeof(float), cudaMemcpyHostToDevice);

    float *d_I_log;
    cudaMalloc(&d_I_log, H * W * sizeof(float));

    jacobiSolver(H, W, d_divG, iterations, tolerance, checkFrequency, d_I_log);
    cudaDeviceSynchronize();

    float *h_I_log;
    cudaMallocHost(&h_I_log, H * W * sizeof(float));
    cudaMemcpy(h_I_log, d_I_log, H * W * sizeof(float), cudaMemcpyDeviceToHost);

    float error = 0.0;
    for (int i = 0; i < H * W; ++i) error += fabs(h_I_log[i] - function2d._f[i]);
    error /= H * W;
    std::cout << "Error: " << error << std::endl;

    cudaFree(d_divG);
    cudaFree(d_I_log);
    
    delete[] h_I_log;

    return 0;
}