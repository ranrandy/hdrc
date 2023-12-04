#include "solvers.h"

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32

/*
    Kernels for computing the residual/error between the previous and current iteration results.
*/
__global__ void errorKernel(const float* current, const float* previous, float* partialSums, const int N) {
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

__global__ void errorReductionKernel(const float* partialSums, float* result, const int numSums) {
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
    Kernel for Jacobi method.
*/
__global__ void jacobiKernel(const float *current, float *result, const float* b, const int W, const int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= W || j >= H) return;

    int idx = j * W + i;

    float up = (j > 0) ? current[idx - W] : current[idx];
    float bottom = (j < H - 1) ? current[idx + W] : current[idx];
    float right = (i < W - 1) ? current[idx + 1] : current[idx];
    float left = (i > 0) ? current[idx - 1] : current[idx];

    result[idx] = 0.25 * (up + bottom + right + left - b[idx]);
}

void jacobiSolver(
    const int H, const int W, 
    const float* d_divG, 
    const int iterations, const float tolerance, const int checkFrequency,
    float* d_I_log)
{
    const int N = H * W;

    dim3 nthreadsJacobi(16, 16, 1);
    dim3 nblocksJacobi((W + nthreadsJacobi.x - 1) / nthreadsJacobi.x, (H + nthreadsJacobi.y - 1) / nthreadsJacobi.y, 1);

    float *d_current;
    cudaMalloc(&d_current, N * sizeof(float));
    cudaMemset(d_current, 0.0, N * sizeof(float));

    int i = 0;
    if (tolerance < 0.0)
    { 
        for (; i < iterations; ) 
        {
            jacobiKernel<<<nblocksJacobi, nthreadsJacobi>>>(d_current, d_I_log, d_divG, W, H);
            cudaDeviceSynchronize();
            std::swap(d_current, d_I_log); ++i;
        }
    } 
    else 
    {   
        int nthreads = 1024;
        int nblocksError = (N + nthreads - 1) / nthreads;
        float *partialErrorSums;
        cudaMalloc(&partialErrorSums, nblocksError * sizeof(float));
        cudaMemset(partialErrorSums, 0.0, nblocksError * sizeof(float));

        int nblocksError2 = (nblocksError + nthreads - 1) / nthreads;
        float* partialErrorSums2;
        cudaMalloc(&partialErrorSums2, nblocksError2 * sizeof(float)); // This should be enough for 2^30 elements/pixels
        cudaMemset(partialErrorSums2, 0.0, nblocksError2 * sizeof(float));

        float error_h;
        float *error_d;
        cudaMalloc(&error_d, sizeof(float));

        for (; i < iterations; ) 
        {
            jacobiKernel<<<nblocksJacobi, nthreadsJacobi>>>(d_current, d_I_log, d_divG, W, H);
            cudaDeviceSynchronize();
            std::swap(d_current, d_I_log); ++i;

            if (i % checkFrequency == 0) // This error calculation may be inefficient. Possibly be improved in the future.
            {   
                errorKernel<<<nblocksError, nthreads>>>(d_current, d_I_log, partialErrorSums, N);
                if (nblocksError > nthreads)
                {
                    errorReductionKernel<<<nblocksError2, nthreads>>>(partialErrorSums, partialErrorSums2, nblocksError);
                    errorReductionKernel<<<1, nthreads>>>(partialErrorSums2, error_d, nblocksError2);
                }
                else
                {
                    errorReductionKernel<<<1, nthreads>>>(partialErrorSums, error_d, nblocksError);
                }
                cudaDeviceSynchronize();
                cudaMemcpy(&error_h, error_d, sizeof(float), cudaMemcpyDeviceToHost);
                
                error_h /= N;

                if (error_h < tolerance) break;
            }
        }
        cudaFree(partialErrorSums);
        cudaFree(partialErrorSums2);
        cudaFree(error_d);
    }

    if (i % 2 == 1) std::swap(d_current, d_I_log);
    cudaFree(d_current);
}


/*
    No kernel for the Gauss-Seidel method because it's sequential.
*/
void gaussSeidelSolver(
    const int H, const int W, 
    const float* d_divG,
    const int iterations, const float tolerance, const int checkFrequency,
    float* d_I_log)
{
    const int N = H * W;

    float *d_current;
    cudaMalloc(&d_current, N * sizeof(float));
    cudaMemset(d_current, 0.0, N * sizeof(float));

    int i = 0;
    if (tolerance < 0.0)
    {
        for (; i < iterations; )
        {
            for (int idx = 0; idx < H * W; ++idx)
            {
                d_I_log[idx] = 0.25 * (d_I_log[idx - 1] + d_I_log[idx - W] + d_current[idx + 1] + d_current[idx + W] - d_divG[idx]);
            }
            std::swap(d_current, d_I_log); ++i;
        }
    }
    else
    {
        int nthreads = 1024;
        int nblocksError = (N + nthreads - 1) / nthreads;
        float *partialErrorSums;
        cudaMalloc(&partialErrorSums, nblocksError * sizeof(float));
        cudaMemset(partialErrorSums, 0.0, nblocksError * sizeof(float));

        int nblocksError2 = (nblocksError + nthreads - 1) / nthreads;
        float* partialErrorSums2;
        cudaMalloc(&partialErrorSums2, nblocksError2 * sizeof(float)); // This should be enough for 2^30 elements/pixels
        cudaMemset(partialErrorSums2, 0.0, nblocksError2 * sizeof(float));

        float error_h;
        float *error_d;
        cudaMalloc(&error_d, sizeof(float));

        for (; i < iterations; )
        {
            for (int idx = 0; idx < H * W; ++idx)
            {
                d_I_log[idx] = 0.25 * (d_I_log[idx - 1] + d_I_log[idx - W] + d_current[idx + 1] + d_current[idx + W] - d_divG[idx]);
            }
            std::swap(d_current, d_I_log); ++i;

            if (i % checkFrequency == 0)
            {
                errorKernel<<<nblocksError, nthreads>>>(d_current, d_I_log, partialErrorSums, N);
                if (nblocksError > nthreads)
                {
                    errorReductionKernel<<<nblocksError2, nthreads>>>(partialErrorSums, partialErrorSums2, nblocksError);
                    errorReductionKernel<<<1, nthreads>>>(partialErrorSums2, error_d, nblocksError2);
                }
                else
                {
                    errorReductionKernel<<<1, nthreads>>>(partialErrorSums, error_d, nblocksError);
                }
                cudaDeviceSynchronize();
                cudaMemcpy(&error_h, error_d, sizeof(float), cudaMemcpyDeviceToHost);
                
                error_h /= N;

                if (error_h < tolerance) break;
            }
        }
        cudaFree(partialErrorSums);
        cudaFree(partialErrorSums2);
        cudaFree(error_d);
    }
    
    if (i % 2 == 1) std::swap(d_current, d_I_log);
    cudaFree(d_current);
}


/*
    Kernel for Red-Black Gauss-Seildel method.
*/
__global__ void redGaussSeidelKernel(const float *current, float *result, const float* b, const int W, const int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= W || j >= H) return;

    int idx = j * W + i;

    if ((i + j) % 2 == 0)
    {
        float up = (j > 0) ? current[idx - W] : current[idx];
        float left = (i > 0) ? current[idx - 1] : current[idx];
        float right = (i < W - 1) ? current[idx + 1] : current[idx];
        float bottom = (j < H - 1) ? current[idx + W] : current[idx];
        result[idx] = 0.25 * (left + up + bottom + right - b[idx]);
    }
}

__global__ void blackGaussSeidelKernel(float *result, const float* b, const int W, const int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= W || j >= H) return;

    int idx = j * W + i;

    if ((i + j) % 2 == 1)
    {
        float up = (j > 0) ? result[idx - W] : result[idx];
        float left = (i > 0) ? result[idx - 1] : result[idx];
        float right = (i < W - 1) ? result[idx + 1] : result[idx];
        float bottom = (j < H - 1) ? result[idx + W] : result[idx];
        result[idx] = 0.25 * (left + up + bottom + right - b[idx]);
    }
}

void gaussSeidelRedBlackSolver(
    const int H, const int W, 
    const float* d_divG,
    const int iterations, const float tolerance, const int checkFrequency,
    float* d_I_log)
{
    const int N = H * W;

    dim3 nthreadsJacobi(16, 16, 1);
    dim3 nblocksJacobi((W + nthreadsJacobi.x - 1) / nthreadsJacobi.x, (H + nthreadsJacobi.y - 1) / nthreadsJacobi.y, 1);

    float *d_current;
    cudaMalloc(&d_current, N * sizeof(float));
    cudaMemset(d_current, 0.0, N * sizeof(float));

    int i = 0;
    if (tolerance < 0.0)
    { 
        for (; i < iterations; ) 
        {
            redGaussSeidelKernel<<<nblocksJacobi, nthreadsJacobi>>>(d_current, d_I_log, d_divG, W, H);
            blackGaussSeidelKernel<<<nblocksJacobi, nthreadsJacobi>>>(d_I_log, d_divG, W, H);
            cudaDeviceSynchronize();
            std::swap(d_current, d_I_log); ++i;
        }
    } 
    else 
    {   
        int nthreads = 1024;
        int nblocksError = (N + nthreads - 1) / nthreads;
        float *partialErrorSums;
        cudaMalloc(&partialErrorSums, nblocksError * sizeof(float));
        cudaMemset(partialErrorSums, 0.0, nblocksError * sizeof(float));

        int nblocksError2 = (nblocksError + nthreads - 1) / nthreads;
        float* partialErrorSums2;
        cudaMalloc(&partialErrorSums2, nblocksError2 * sizeof(float)); // This should be enough for 2^30 elements/pixels
        cudaMemset(partialErrorSums2, 0.0, nblocksError2 * sizeof(float));

        float error_h;
        float *error_d;
        cudaMalloc(&error_d, sizeof(float));

        for (; i < iterations; ) 
        {
            redGaussSeidelKernel<<<nblocksJacobi, nthreadsJacobi>>>(d_current, d_I_log, d_divG, W, H);
            blackGaussSeidelKernel<<<nblocksJacobi, nthreadsJacobi>>>(d_I_log, d_divG, W, H);
            cudaDeviceSynchronize();
            std::swap(d_current, d_I_log); ++i;

            if (i % checkFrequency == 0) // This error calculation may be inefficient. Possibly be improved in the future.
            {   
                errorKernel<<<nblocksError, nthreads>>>(d_current, d_I_log, partialErrorSums, N);
                if (nblocksError > nthreads)
                {
                    errorReductionKernel<<<nblocksError2, nthreads>>>(partialErrorSums, partialErrorSums2, nblocksError);
                    errorReductionKernel<<<1, nthreads>>>(partialErrorSums2, error_d, nblocksError2);
                }
                else
                {
                    errorReductionKernel<<<1, nthreads>>>(partialErrorSums, error_d, nblocksError);
                }
                cudaDeviceSynchronize();
                cudaMemcpy(&error_h, error_d, sizeof(float), cudaMemcpyDeviceToHost);
                
                error_h /= N;

                if (error_h < tolerance) break;
            }
        }
        cudaFree(partialErrorSums);
        cudaFree(partialErrorSums2);
        cudaFree(error_d);
    }

    if (i % 2 == 1) std::swap(d_current, d_I_log);
    cudaFree(d_current);
}


/*
    Kernel for Red-Black Gauss-Seildel method with overrelaxation.
*/
__global__ void redGaussSeidelSORKernel(const float *current, float *result, const float* b, const int W, const int H, const float omega) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= W || j >= H) return;

    int idx = j * W + i;

    if ((i + j) % 2 == 0)
    {
        float up = (j > 0) ? current[idx - W] : current[idx];
        float left = (i > 0) ? current[idx - 1] : current[idx];
        float right = (i < W - 1) ? current[idx + 1] : current[idx];
        float bottom = (j < H - 1) ? current[idx + W] : current[idx];
        result[idx] = 0.25 * (omega * (left + up) + bottom + right - b[idx]);
    }
}

__global__ void blackGaussSeidelSORKernel(float *result, const float* b, const int W, const int H, const float omega) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= W || j >= H) return;

    int idx = j * W + i;

    if ((i + j) % 2 == 1)
    {
        float up = (j > 0) ? result[idx - W] : result[idx];
        float left = (i > 0) ? result[idx - 1] : result[idx];
        float right = (i < W - 1) ? result[idx + 1] : result[idx];
        float bottom = (j < H - 1) ? result[idx + W] : result[idx];
        result[idx] = 0.25 * (omega * (left + up) + bottom + right - b[idx]);
    }
}

void gaussSeidelRedBlackSORSolver(
    const int H, const int W, 
    const float* d_divG,
    const int iterations, const float tolerance, const int checkFrequency, const float omega,
    float* d_I_log)
{
    const int N = H * W;

    dim3 nthreadsJacobi(16, 16, 1);
    dim3 nblocksJacobi((W + nthreadsJacobi.x - 1) / nthreadsJacobi.x, (H + nthreadsJacobi.y - 1) / nthreadsJacobi.y, 1);

    float *d_current;
    cudaMalloc(&d_current, N * sizeof(float));
    cudaMemset(d_current, 0.0, N * sizeof(float));

    int i = 0;
    if (tolerance < 0.0)
    { 
        for (; i < iterations; ) 
        {
            redGaussSeidelSORKernel<<<nblocksJacobi, nthreadsJacobi>>>(d_current, d_I_log, d_divG, W, H, omega);
            blackGaussSeidelSORKernel<<<nblocksJacobi, nthreadsJacobi>>>(d_I_log, d_divG, W, H, omega);
            cudaDeviceSynchronize();
            std::swap(d_current, d_I_log); ++i;
        }
    } 
    else 
    {   
        int nthreads = 1024;
        int nblocksError = (N + nthreads - 1) / nthreads;
        float *partialErrorSums;
        cudaMalloc(&partialErrorSums, nblocksError * sizeof(float));
        cudaMemset(partialErrorSums, 0.0, nblocksError * sizeof(float));

        int nblocksError2 = (nblocksError + nthreads - 1) / nthreads;
        float* partialErrorSums2;
        cudaMalloc(&partialErrorSums2, nblocksError2 * sizeof(float)); // This should be enough for 2^30 elements/pixels
        cudaMemset(partialErrorSums2, 0.0, nblocksError2 * sizeof(float));

        float error_h;
        float *error_d;
        cudaMalloc(&error_d, sizeof(float));

        for (; i < iterations; ) 
        {
            redGaussSeidelSORKernel<<<nblocksJacobi, nthreadsJacobi>>>(d_current, d_I_log, d_divG, W, H, omega);
            blackGaussSeidelSORKernel<<<nblocksJacobi, nthreadsJacobi>>>(d_I_log, d_divG, W, H, omega);
            cudaDeviceSynchronize();
            std::swap(d_current, d_I_log); ++i;

            if (i % checkFrequency == 0) // This error calculation may be inefficient. Possibly be improved in the future.
            {   
                errorKernel<<<nblocksError, nthreads>>>(d_current, d_I_log, partialErrorSums, N);
                if (nblocksError > nthreads)
                {
                    errorReductionKernel<<<nblocksError2, nthreads>>>(partialErrorSums, partialErrorSums2, nblocksError);
                    errorReductionKernel<<<1, nthreads>>>(partialErrorSums2, error_d, nblocksError2);
                }
                else
                {
                    errorReductionKernel<<<1, nthreads>>>(partialErrorSums, error_d, nblocksError);
                }
                cudaDeviceSynchronize();
                cudaMemcpy(&error_h, error_d, sizeof(float), cudaMemcpyDeviceToHost);
                
                error_h /= N;

                if (error_h < tolerance) break;
            }
        }
        cudaFree(partialErrorSums);
        cudaFree(partialErrorSums2);
        cudaFree(error_d);
    }

    if (i % 2 == 1) std::swap(d_current, d_I_log);
    cudaFree(d_current);
}


void fullMultigridSolver(    
    const int H, const int W, 
    const float* d_divG, 
    const int iterations, const float tolerence,
    float* d_I_log)
{

}
