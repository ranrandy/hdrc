#include "solvers.h"
#include "solvers.cu"


/*
    This reduction of sum of errors may be more efficient in some cases. 
    But generally this makes no difference with using AtmoicAdd in this application
*/
__global__ void blockErrorsKernel(const float* current, const float* previous, float* partialSums, const int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float s_sum[1024 / WARP_SIZE];

    int nwarps = blockDim.x / WARP_SIZE;
    int my_warp = threadIdx.x / WARP_SIZE;

    float sum = 0.0;
    
    if (tid < N)
        sum = fabsf(current[tid] - previous[tid]);
    __syncwarp();

    // shift within warp
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) 
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    
    // sum over warps if needed
    if (nwarps > 1) {
        if (threadIdx.x % WARP_SIZE == 0)
            s_sum[my_warp] = sum;
        __syncthreads();

        if (threadIdx.x == 0) {
            for (int i = 1; i < nwarps; ++i)
                sum += s_sum[i];
        }
    }

    // final step - store results into the main device memory
    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = sum;
    }
}

__global__ void blockErrorsReductionKernel(const float* partialSums, float* result, const int numSums) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float s_sum[1024 / WARP_SIZE];
    
    int nwarps = blockDim.x / WARP_SIZE;
    int my_warp = threadIdx.x / WARP_SIZE;

    float sum = 0.0;
    
    if (tid < numSums)
        sum = partialSums[tid];
    __syncwarp();

    // shift within warp
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) 
        sum += __shfl_down_sync(FULL_MASK, sum, offset);

    // sum over warps if needed
    if (nwarps > 1) {
        if (threadIdx.x % WARP_SIZE == 0)
            s_sum[my_warp] = sum;
        __syncthreads();

        if (threadIdx.x == 0) {
            for (int i = 1; i < nwarps; ++i)
                sum += s_sum[i];
        }
    }

    // final step - store results into the main device memory
    if (threadIdx.x == 0) {
        result[blockIdx.x] = sum;
    }
}

/*
    The solver that uses reduction of sum to calculate residual/error, which is not necessary.
*/
int solver(
    const int H, const int W, 
    const float* d_divG, const int method,
    const int iterations, const float tolerance, const int checkFrequency,
    float* d_I_log)
{
    const int N = H * W;

    dim3 nthreadsMethod(16, 16, 1);
    dim3 nblocksMethod((W + nthreadsMethod.x - 1) / nthreadsMethod.x, (H + nthreadsMethod.y - 1) / nthreadsMethod.y, 1);

    float *d_current;
    cudaMalloc(&d_current, N * sizeof(float));
    cudaMemset(d_current, 0.0, N * sizeof(float));
    cudaMemset(d_I_log, 0.0, N * sizeof(float));

    methodFunction methodKernel;
    switch (method) {
        case 0:
            methodKernel = jacobi;
            break;
        case 1:
            methodKernel = gaussSeidelRedBlack;
            break;
        case 2:
            methodKernel = gaussSeidelRedBlackSOR;
            break;
        default:
            return -1;
    }

    int i = 0;

    int nblocksError = (N + MAX_THREADS - 1) / MAX_THREADS;

    float *partialErrorSums;
    cudaMalloc(&partialErrorSums, nblocksError * sizeof(float));
    cudaMemset(partialErrorSums, 0.0, nblocksError * sizeof(float));

    int nblocksError2 = (nblocksError + MAX_THREADS - 1) / MAX_THREADS;
    float* partialErrorSums2;
    cudaMalloc(&partialErrorSums2, nblocksError2 * sizeof(float)); // This should be enough for 2^30 elements/pixels
    cudaMemset(partialErrorSums2, 0.0, nblocksError2 * sizeof(float));

    float error_h;
    float *error_d;
    cudaMalloc(&error_d, sizeof(float));

    for (; i < iterations; ) 
    {
        methodKernel(H, W, d_divG, nblocksMethod, nthreadsMethod, d_current, d_I_log);
        cudaDeviceSynchronize();
        std::swap(d_current, d_I_log); ++i;

        if (i % checkFrequency == 0) // This error calculation may be inefficient. Possibly be improved in the future.
        {   
            cudaMemset(error_d, 0.0, sizeof(float));

            blockErrorsKernel<<<nblocksError, MAX_THREADS>>>(d_current, d_I_log, partialErrorSums, N);
            if (nblocksError > MAX_THREADS)
            {
                blockErrorsReductionKernel<<<nblocksError2, MAX_THREADS>>>(partialErrorSums, partialErrorSums2, nblocksError);
                blockErrorsReductionKernel<<<1, MAX_THREADS>>>(partialErrorSums2, error_d, nblocksError2);
            }
            else
            {
                blockErrorsReductionKernel<<<1, MAX_THREADS>>>(partialErrorSums, error_d, nblocksError);
            }
            
            cudaMemcpy(&error_h, error_d, sizeof(float), cudaMemcpyDeviceToHost);
            
            error_h /= N;

            if (error_h < tolerance) break;
        }
    }
    cudaFree(partialErrorSums);
    cudaFree(partialErrorSums2);
    cudaFree(error_d);

    if (i % 2 == 1) std::swap(d_current, d_I_log);
    cudaFree(d_current);
    return i;
}


void debug(float* d_I_log)
{
    float *test;
    cudaMallocHost(&test, 25 * sizeof(float));

    cudaMemcpy(test, &d_I_log[812 * 1600 + 400], 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(test + 5, &d_I_log[813 * 1600 + 400], 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(test + 10, &d_I_log[814 * 1600 + 400], 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(test + 15, &d_I_log[815 * 1600 + 400], 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(test + 20, &d_I_log[816 * 1600 + 400], 5 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    
    std::cout << "12-16, 400-404" << std::endl;

    for (int r = 0; r < 5; ++r) {
        for (int c = 0; c < 5; ++c) {
            std::cout << test[r * 5 + c] << " ";
        }
        std::cout << std::endl;
    }

    cudaFreeHost(test);
}