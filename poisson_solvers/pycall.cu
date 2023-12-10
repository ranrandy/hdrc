#include "pycall.h"
#include "solvers.h"

int solve(
    const int H, const int W,
    const torch::Tensor& d_div_G, 
    const int method, const std::vector<float>& args,
    const int iterations, const int checkFrequency, const float tolerance,
    torch::Tensor& h_I_log)
{
    float* arguments;

    float *d_I_log;
    cudaMalloc(&d_I_log, H * W * sizeof(float));
    cudaMemset(d_I_log, 0.0, H * W * sizeof(float));

    int iter_converge = 0;

    std::cout << "method: " << method << std::endl;

    // Set up timer
    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Start timer
    cudaEventRecord(start);
    
    if (method <= 4) // Simple Solver
    {
        // std::cout << "iterations: " << iterations << std::endl; 
        // std::cout << "checkFrequency: " << checkFrequency << std::endl; 
        // std::cout << "tolerance: " << tolerance << std::endl; 

        assert (args.size() == 0 || args.size() == 1);

        cudaMallocHost(&arguments, 1 * sizeof(float));
        arguments[0] = (args.size() == 1) ? args[0] : 1.90;

        // std::cout << "omega: " << arguments[0] << std::endl;
        
        iter_converge = simpleSolver(
            H, W, d_div_G.contiguous().data<float>(),
            method, arguments,
            iterations, checkFrequency, tolerance,
            d_I_log);
    }
    else // Multigrid Solver
    {
        // std::cout << "cycleIterations: " << iterations << std::endl; 
        // std::cout << "checkCycleFrequency: " << checkFrequency << std::endl; 
        // std::cout << "cycleTolerance: " << tolerance << std::endl; 

        assert (args.size() == 6 || args.size() == 7);

        cudaMallocHost(&arguments, 7 * sizeof(float));
        for (int i = 0; i < args.size(); ++i) arguments[i] = args[i];
        arguments[6] = (args.size() == 7) ? args[0] : 1.90;

        // std::cout << "multigridSmoothingMethod: " << arguments[0] << std::endl; 
        // std::cout << "prepostSmoothingIterations: " << arguments[1] << std::endl; 
        // std::cout << std::endl; 
        // std::cout << "coarsestSideLength: " << arguments[2] << std::endl; 
        // std::cout << std::endl; 
        // std::cout << "multigridCoarsestIterations: " << arguments[3] << std::endl; 
        // std::cout << "checkCoarsestFrequency: " << arguments[4] << std::endl; 
        // std::cout << "CoarsestTolerance: " << arguments[5] << std::endl; 
        // std::cout << std::endl; 
        // std::cout << "omega: " << arguments[6] << std::endl;

        iter_converge = multigridSolver(
            H, W, d_div_G.contiguous().data<float>(),
            method, arguments,
            iterations, checkFrequency, tolerance,
            d_I_log);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_I_log.contiguous().data<float>(), d_I_log, H * W * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Stop timer
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time duration: " << std::setprecision (6) << milliseconds << " milliseconds" << std::endl;
    std::cout << std::endl;

    cudaFree(d_I_log);
    cudaFreeHost(arguments);

    return iter_converge;
}