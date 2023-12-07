#include <iostream>
#include <string>

#include "solvers.h"
#include "debug_function2D.h"


int main(int argc, char* argv[]) {
    int method = std::stoi(argv[1]), 
        iterations = std::stoi(argv[2]), 
        checkFrequency = std::stoi(argv[3]), 
        warmup = std::stoi(argv[5]),
        measure = std::stoi(argv[6]); 
    float tolerance = std::stof(argv[4]);
    float omega = (argc == 8) ? std::stof(argv[7]) : 1.45;

    std::cout << std::endl; 
    std::cout << "method: " << method << std::endl; 
    std::cout << "iterations: " << iterations << std::endl; 
    std::cout << "checkFrequency: " << checkFrequency << std::endl; 
    std::cout << "tolerance: " << tolerance << std::endl; 
    std::cout << std::endl; 
    std::cout << "warmup: " << warmup << std::endl; 
    std::cout << "measure: " << measure << std::endl;
    std::cout << "omega: " << omega << std::endl;
    std::cout << std::endl; 
    
    // --------------------------------------------- 2D ---------------------------------------------
    int W = 1600, H = 1200;
    Function2D function2d(W, H);

    int j = 9, i = 25;
    float real_laplacian = function2d._real_f_laplacian[j * W + i];
    float estimate_laplacian = function2d._estimated_f_laplacian[j * W + i];

    std::cout << "Function: f(x) = sin(pi*(x+y)/100.0): " << std::endl;
    std::cout << "L1 Estimated Laplacian Error " << std::setprecision (6) << fabs(estimate_laplacian - real_laplacian) << std::endl;
    std::cout << std::endl;

    // ---------------------------------- Solve Poisson Equation ------------------------------------
    float *d_divG;
    cudaMalloc(&d_divG, H * W * sizeof(float));
    cudaMemcpy(d_divG, function2d._estimated_f_laplacian, H * W * sizeof(float), cudaMemcpyHostToDevice);

    float *d_I_log;
    cudaMalloc(&d_I_log, H * W * sizeof(float));

    float *h_I_log;
    cudaMallocHost(&h_I_log, H * W * sizeof(float));

    int iter_converge = 0;

    float *args;
    cudaMallocHost(&args, sizeof(float));
    args[0] = omega;

    // Warm up
    for (int iter = 0; iter < warmup; ++iter)
    {
        iter_converge = simpleSolver(H, W, d_divG, method, args, iterations, tolerance, checkFrequency, d_I_log);
        cudaDeviceSynchronize();

        cudaMemcpy(h_I_log, d_I_log, H * W * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

    // Set up timer
    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timer
    cudaEventRecord(start);

    for (int iter = 0; iter < measure; ++iter)
    {
        iter_converge = simpleSolver(H, W, d_divG, method, args, iterations, tolerance, checkFrequency, d_I_log);
        cudaDeviceSynchronize();

        cudaMemcpy(h_I_log, d_I_log, H * W * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

    // Stop timer
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time duration: " << std::setprecision (6) << milliseconds / measure << " milliseconds" << std::endl;
    std::cout << std::endl;

    // --------------------------------------- Calculate Error --------------------------------------
    float error = 0.0;
    for (int i = 0; i < H * W; ++i) {
        error += fabs(h_I_log[i] - function2d._f[i]);
    }
    error /= H * W;
    std::cout << "Error: " << std::setprecision (6) << error << std::endl;
    std::cout << "Number of iteration until convergence: " << std::setprecision (6) << iter_converge << std::endl;
    std::cout << std::endl;

    // --------------------------------------- Free Memory --------------------------------------
    cudaFree(d_divG);
    cudaFree(d_I_log);
    cudaFreeHost(h_I_log);
    return 0;
}