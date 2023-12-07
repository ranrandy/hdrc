#include "solvers.h"


/*
    Computing the residual/error between the previous and current iteration results. In most cases, atomicAdd should be fast enough.
*/
__global__ void atomicAddBlockErrorsKernel(const float* current, const float* previous, float* result, const int N) {
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
        atomicAdd(result, sum);
    }
}

/*
    0. Jacobi method.
*/
__global__ void jacobiKernel(const int H, const int W, const float* b, const float *current, float *result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= W || j >= H) return;

    int idx = j * W + i;

    float up = (j > 0) ? current[idx - W] : current[idx];
    float left = (i > 0) ? current[idx - 1] : current[idx];
    float right = (i < W - 1) ? current[idx + 1] : current[idx];
    float bottom = (j < H - 1) ? current[idx + W] : current[idx];

    result[idx] = 0.25f * (up + left + right + bottom - b[idx]);
}

void jacobi(const int H, const int W, const float* d_divG, const dim3 nblocks, const dim3 nthreads, float* d_current, float* d_result, const float* args)
{
    jacobiKernel<<<nblocks, nthreads>>>(H, W, d_divG, d_current, d_result);
}


/*
    1. Red-Black Gauss-Seildel method.
*/
__global__ void redGaussSeidelKernel(const int H, const int W, const float* b, const float *current, float *result) {
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
        result[idx] = 0.25f * (up + left + right + bottom - b[idx]);
    }
}

__global__ void blackGaussSeidelKernel(const int H, const int W, const float* b, float *result) {
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
        result[idx] = 0.25f * (up + left + right + bottom - b[idx]);
    }
}

void gaussSeidelRedBlack(const int H, const int W, const float* d_divG, const dim3 nblocks, const dim3 nthreads, float* d_current, float* d_result, const float* args)
{
    redGaussSeidelKernel<<<nblocks, nthreads>>>(H, W, d_divG, d_current, d_result);
    blackGaussSeidelKernel<<<nblocks, nthreads>>>(H, W, d_divG, d_result);
}


/*
    2. Red-Black Gauss-Seildel method with successive over relaxation (SOR).
*/
__global__ void redGaussSeidelSORKernel(const int H, const int W, const float* b, const float *current, float *result, const float w_opt) {
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
        result[idx] = (1 - w_opt) * current[idx] + w_opt * 0.25f * (up + left + right + bottom - b[idx]);
    }
}

__global__ void blackGaussSeidelSORKernel(const int H, const int W, const float* b, const float *current, float *result, const float w_opt) {
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
        result[idx] = (1 - w_opt) * current[idx] + w_opt * 0.25f * (up + left + right + bottom - b[idx]);
    }
}

void gaussSeidelRedBlackSOR(const int H, const int W, const float* d_divG, const dim3 nblocks, const dim3 nthreads, float* d_current, float* d_result, const float* args)
{
    // float w_opt = 2.0f / (1.0f + sinf(M_PI / max(H, W)));
    redGaussSeidelSORKernel<<<nblocks, nthreads>>>(H, W, d_divG, d_current, d_result, args[0]);
    blackGaussSeidelSORKernel<<<nblocks, nthreads>>>(H, W, d_divG, d_current, d_result, args[0]);
}


/*
    --> Red-Black pre-reordering of grids.
*/
__global__ void fillInRedBlackBGaussSeidel2Kernel(const int H, const int W, const float* b, float* red_black_b, const int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= W || j >= H) return;

    int idx = j * W + i;
    int squeezed_j;

    if ((i+j) % 2 == 0) // red
    {
        if (i % 2 == 1) squeezed_j = (j - 1) / 2;
        else squeezed_j = j / 2;
        int squeezed_idx = squeezed_j * W + i; 
        red_black_b[squeezed_idx] = b[idx];
    }
    else // black
    {
        if (i % 2 == 1) squeezed_j = j / 2;
        else squeezed_j = (j - 1) / 2; 
        int squeezed_idx = squeezed_j * W + i; 
        red_black_b[squeezed_idx + offset] = b[idx];
    }
}

__global__ void fillInGaussSeidel2Kernel(const int H, const int W, const float* red, const float* black, float* result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= W || j >= H) return;

    int idx = j * W + i;
    int squeezed_j;

    if ((j + i) % 2 == 0) // red
    {
        if (i % 2 == 1) squeezed_j = (j - 1) / 2;
        else squeezed_j = j / 2;
        int squeezed_idx = squeezed_j * W + i; 
        result[idx] = red[squeezed_idx];
    }
    else // black
    {
        if (i % 2 == 1) squeezed_j = j / 2;
        else squeezed_j = (j - 1) / 2; 
        int squeezed_idx = squeezed_j * W + i; 
        result[idx] = black[squeezed_idx];
    }
}

/*
    3. Gauss-Seidel with pre-reordering of grids.
*/
__global__ void redGaussSeidelKernel2(const int H, const int W, const float* b, const float *black, float *red) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= W || j >= H) return;

    int idx = j * W + i;

    if (i % 2 == 1)
    {
        int orig_j = j * 2 + 1;
        if (orig_j >= H) return;

        float up = black[idx];
        float left = black[idx - 1];
        float right = (i < W - 1) ? black[idx + 1] : red[idx];
        float bottom = (orig_j < H - 1) ? black[idx + W] : red[idx];
        red[idx] = 0.25f * (up + left + right + bottom - b[idx]);
    }
    else
    {
        int orig_j = j * 2;
        if (orig_j >= H) return;

        float up = (orig_j > 0) ? black[idx - W] : red[idx];
        float left = (i > 0) ? black[idx - 1] : red[idx];
        float right = (i < W - 1) ? black[idx + 1] : red[idx];
        float bottom = (orig_j < H - 1) ? black[idx] : red[idx];
        red[idx] = 0.25f * (up + left + right + bottom - b[idx]);
    }
}

__global__ void blackGaussSeidelKernel2(const int H, const int W, const float* b, const float *red, float *black, const int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= W) return;

    int idx = j * W + i;

    if (i % 2 == 1)
    {
        int orig_j = j * 2;
        if (orig_j >= H) return;

        float up = (orig_j > 0) ? red[idx - W] : black[idx];
        float left = red[idx - 1];
        float right = (i < W - 1) ? red[idx + 1] : black[idx];
        float bottom = (orig_j < H - 1) ? red[idx] : black[idx];
        black[idx] = 0.25f * (up + left + right + bottom - b[idx + offset]);
    }
    else
    {
        int orig_j = j * 2 + 1;
        if (orig_j >= H) return;

        float up = red[idx];
        float left = (i > 0) ? red[idx - 1] : black[idx];
        float right = (i < W - 1) ? red[idx + 1] : black[idx];
        float bottom = (orig_j < H - 1) ? red[idx + W] : black[idx];
        black[idx] = 0.25f * (up + left + right + bottom - b[idx + offset]);
    }
}

void gaussSeidelRedBlack2(const int H, const int W, const float* d_divG, const dim3 nblocks, const dim3 nthreads, float* red, float* black, const float* args)
{
    redGaussSeidelKernel2<<<nblocks, nthreads>>>(H, W, d_divG, black, red);
    blackGaussSeidelKernel2<<<nblocks, nthreads>>>(H, W, d_divG, red, black, (H * W) / 2);
}

/*
    4. Gauss-Seidel with SOR and pre-reordering of grids.
*/
__global__ void redGaussSeidelKernel2SOR(const int H, const int W, const float* b, const float *black, float *red, const float w_opt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= W || j >= H) return;

    int idx = j * W + i;

    if (i % 2 == 1)
    {
        int orig_j = j * 2 + 1;
        if (orig_j >= H) return;

        float up = black[idx];
        float left = black[idx - 1];
        float right = (i < W - 1) ? black[idx + 1] : red[idx];
        float bottom = (orig_j < H - 1) ? black[idx + W] : red[idx];
        red[idx] = (1 - w_opt) * red[idx] + w_opt * 0.25f * (up + left + right + bottom - b[idx]);
    }
    else
    {
        int orig_j = j * 2;
        if (orig_j >= H) return;

        float up = (orig_j > 0) ? black[idx - W] : red[idx];
        float left = (i > 0) ? black[idx - 1] : red[idx];
        float right = (i < W - 1) ? black[idx + 1] : red[idx];
        float bottom = (orig_j < H - 1) ? black[idx] : red[idx];
        red[idx] = (1 - w_opt) * red[idx] + w_opt * 0.25f * (up + left + right + bottom - b[idx]);
    }
}

__global__ void blackGaussSeidelKernel2SOR(const int H, const int W, const float* b, const float *red, float *black, const int offset, const float w_opt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= W) return;

    int idx = j * W + i;

    if (i % 2 == 1)
    {
        int orig_j = j * 2;
        if (orig_j >= H) return;

        float up = (orig_j > 0) ? red[idx - W] : black[idx];
        float left = red[idx - 1];
        float right = (i < W - 1) ? red[idx + 1] : black[idx];
        float bottom = (orig_j < H - 1) ? red[idx] : black[idx];
        black[idx] = (1 - w_opt) * black[idx] + w_opt * 0.25f * (up + left + right + bottom - b[idx + offset]);
    }
    else
    {
        int orig_j = j * 2 + 1;
        if (orig_j >= H) return;

        float up = red[idx];
        float left = (i > 0) ? red[idx - 1] : black[idx];
        float right = (i < W - 1) ? red[idx + 1] : black[idx];
        float bottom = (orig_j < H - 1) ? red[idx + W] : black[idx];
        black[idx] = (1 - w_opt) * black[idx] + w_opt * 0.25f * (up + left + right + bottom - b[idx + offset]);
    }
}

void gaussSeidelRedBlack2SOR(const int H, const int W, const float* d_divG, const dim3 nblocks, const dim3 nthreads, float* red, float* black, const float* args)
{
    redGaussSeidelKernel2SOR<<<nblocks, nthreads>>>(H, W, d_divG, black, red, args[0]);
    blackGaussSeidelKernel2SOR<<<nblocks, nthreads>>>(H, W, d_divG, red, black, (H * W) / 2, args[0]);
}


int simpleSolver(
    const int H, const int W, 
    const float* d_divG, const int method, const float* args, 
    const int iterations, const float tolerance, const int checkFrequency,
    float* d_I_log)
{
    const int N = H * W, N2 = H * W / 2;

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
        case 3:
            methodKernel = gaussSeidelRedBlack2;
            break;
        case 4:
            methodKernel = gaussSeidelRedBlack2SOR;
            break;
        default:
            return -1;
    }

    int nblocksError = (N + MAX_THREADS - 1) / MAX_THREADS;
    float error_h;
    float *error_d;
    cudaMalloc(&error_d, sizeof(float));

    if (method <= 2) 
    {                
        dim3 nthreadsMethod(16, 16, 1);
        dim3 nblocksMethod((W + nthreadsMethod.x - 1) / nthreadsMethod.x, (H + nthreadsMethod.y - 1) / nthreadsMethod.y, 1);

        float *d_current, *d_result;
        cudaMalloc(&d_current, N * sizeof(float));
        cudaMemset(d_current, 0.0, N * sizeof(float));
        cudaMalloc(&d_result, N * sizeof(float));
        cudaMemset(d_result, 0.0, N * sizeof(float));

        int i = 0;
        for (; i < iterations; ) 
        {
            methodKernel(H, W, d_divG, nblocksMethod, nthreadsMethod, d_current, d_result, args);
            cudaDeviceSynchronize(); 
            std::swap(d_current, d_result); ++i;

            if (i % checkFrequency == 0)
            {   
                cudaMemset(error_d, 0.0, sizeof(float));
                atomicAddBlockErrorsKernel<<<nblocksError, MAX_THREADS>>>(d_current, d_result, error_d, N);
                cudaDeviceSynchronize();
                cudaMemcpy(&error_h, error_d, sizeof(float), cudaMemcpyDeviceToHost);
                error_h /= N;

                if (error_h < tolerance) break;
            }
        }

        if (i % 2 == 1) std::swap(d_current, d_result);
        cudaMemcpy(d_I_log, d_result, N * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaFree(d_current);
        cudaFree(d_result);
        cudaFree(error_d);
        return i;
    }

    if (method > 2) 
    {
        dim3 nthreadsMethod(32, 16, 1);
        dim3 nblocksMethod((W + nthreadsMethod.x - 1) / nthreadsMethod.x, (int(H/2) + nthreadsMethod.y - 1) / nthreadsMethod.y, 1);

        dim3 nthreadsFillIn(16, 16, 1);
        dim3 nblocksFillIn((W + nthreadsFillIn.x - 1) / nthreadsFillIn.x, (H + nthreadsFillIn.y - 1) / nthreadsFillIn.y, 1);

        float *red, *black;
        cudaMalloc(&red, N2 * sizeof(float));
        cudaMemset(red, 0.0, N2 * sizeof(float));
        cudaMalloc(&black, N2 * sizeof(float));
        cudaMemset(black, 0.0, N2 * sizeof(float));

        float *prev_red;
        cudaMalloc(&prev_red, N2 * sizeof(float));

        float *red_black_divG;
        cudaMalloc(&red_black_divG, N * sizeof(float));
        fillInRedBlackBGaussSeidel2Kernel<<<nblocksFillIn, nthreadsFillIn>>>(H, W, d_divG, red_black_divG, N2);

        int i = 0;
        for (; i < iterations; ) 
        {
            if ((i+1) % checkFrequency == 0) cudaMemcpy(prev_red, red, N2 * sizeof(float), cudaMemcpyDeviceToDevice);

            methodKernel(H, W, red_black_divG, nblocksMethod, nthreadsMethod, red, black, args);
            cudaDeviceSynchronize();
            ++i;

            if (i % checkFrequency == 0)
            {   
                cudaMemset(error_d, 0.0, sizeof(float));
                atomicAddBlockErrorsKernel<<<nblocksError, MAX_THREADS>>>(red, prev_red, error_d, N2);
                cudaDeviceSynchronize();
                cudaMemcpy(&error_h, error_d, sizeof(float), cudaMemcpyDeviceToHost);
                error_h /= N2;

                if (error_h < tolerance) break;
            }
        }

        fillInGaussSeidel2Kernel<<<nblocksFillIn, nthreadsFillIn>>>(H, W, red, black, d_I_log);
        cudaDeviceSynchronize();

        cudaFree(red);
        cudaFree(black);
        cudaFree(prev_red);
        cudaFree(red_black_divG);
        cudaFree(error_d);
        return i;
    }

    cudaFree(error_d);
    return -1;
}


void multigridSolver(    
    const int H, const int W, 
    const float* d_divG, 
    const int iterations, const float tolerence,
    float* d_I_log)
{

}
