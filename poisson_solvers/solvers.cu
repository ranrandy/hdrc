#include "solvers.h"


/*
    Computing the error between the previous and current iteration results. In most cases, atomicAdd should be fast enough.
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

    float top = (j > 0) ? current[idx - W] : current[idx];
    float left = (i > 0) ? current[idx - 1] : current[idx];
    float right = (i < W - 1) ? current[idx + 1] : current[idx];
    float bottom = (j < H - 1) ? current[idx + W] : current[idx];

    result[idx] = 0.25f * (top + left + right + bottom - b[idx]);
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
        float top = (j > 0) ? current[idx - W] : current[idx];
        float left = (i > 0) ? current[idx - 1] : current[idx];
        float right = (i < W - 1) ? current[idx + 1] : current[idx];
        float bottom = (j < H - 1) ? current[idx + W] : current[idx];
        result[idx] = 0.25f * (top + left + right + bottom - b[idx]);
    }
}

__global__ void blackGaussSeidelKernel(const int H, const int W, const float* b, float *result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= W || j >= H) return;

    int idx = j * W + i;

    if ((i + j) % 2 == 1)
    {
        float top = (j > 0) ? result[idx - W] : result[idx];
        float left = (i > 0) ? result[idx - 1] : result[idx];
        float right = (i < W - 1) ? result[idx + 1] : result[idx];
        float bottom = (j < H - 1) ? result[idx + W] : result[idx];
        result[idx] = 0.25f * (top + left + right + bottom - b[idx]);
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
        float top = (j > 0) ? current[idx - W] : current[idx];
        float left = (i > 0) ? current[idx - 1] : current[idx];
        float right = (i < W - 1) ? current[idx + 1] : current[idx];
        float bottom = (j < H - 1) ? current[idx + W] : current[idx];
        result[idx] = (1 - w_opt) * current[idx] + w_opt * 0.25f * (top + left + right + bottom - b[idx]);
    }
}

__global__ void blackGaussSeidelSORKernel(const int H, const int W, const float* b, const float *current, float *result, const float w_opt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= W || j >= H) return;

    int idx = j * W + i;

    if ((i + j) % 2 == 1)
    {
        float top = (j > 0) ? result[idx - W] : result[idx];
        float left = (i > 0) ? result[idx - 1] : result[idx];
        float right = (i < W - 1) ? result[idx + 1] : result[idx];
        float bottom = (j < H - 1) ? result[idx + W] : result[idx];
        result[idx] = (1 - w_opt) * current[idx] + w_opt * 0.25f * (top + left + right + bottom - b[idx]);
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
__global__ void fillInRedBlackInitGaussSeidel2Kernel(const int H, const int W, const float* init_guess, float* red, float* black) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= W || j >= H) return;

    int idx = j * W + i;
    int squeezed_j;

    if ((i+j) % 2 == 0) // red
    {
        if (i % 2 == 1) squeezed_j = (j - 1) / 2;
        else squeezed_j = j / 2;
        red[squeezed_j * W + i] = init_guess[idx];
    }
    else // black
    {
        if (i % 2 == 1) squeezed_j = j / 2;
        else squeezed_j = (j - 1) / 2; 
        black[squeezed_j * W + i] = init_guess[idx];
    }
}

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
        red_black_b[squeezed_j * W + i] = b[idx];
    }
    else // black
    {
        if (i % 2 == 1) squeezed_j = j / 2;
        else squeezed_j = (j - 1) / 2; 
        red_black_b[squeezed_j * W + i + offset] = b[idx];
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

        float top = black[idx];
        float left = black[idx - 1];
        float right = (i < W - 1) ? black[idx + 1] : red[idx];
        float bottom = (orig_j < H - 1) ? black[idx + W] : red[idx];
        red[idx] = 0.25f * (top + left + right + bottom - b[idx]);
    }
    else
    {
        int orig_j = j * 2;
        if (orig_j >= H) return;

        float top = (orig_j > 0) ? black[idx - W] : red[idx];
        float left = (i > 0) ? black[idx - 1] : red[idx];
        float right = (i < W - 1) ? black[idx + 1] : red[idx];
        float bottom = (orig_j < H - 1) ? black[idx] : red[idx];
        red[idx] = 0.25f * (top + left + right + bottom - b[idx]);
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

        float top = (orig_j > 0) ? red[idx - W] : black[idx];
        float left = red[idx - 1];
        float right = (i < W - 1) ? red[idx + 1] : black[idx];
        float bottom = (orig_j < H - 1) ? red[idx] : black[idx];
        black[idx] = 0.25f * (top + left + right + bottom - b[idx + offset]);
    }
    else
    {
        int orig_j = j * 2 + 1;
        if (orig_j >= H) return;

        float top = red[idx];
        float left = (i > 0) ? red[idx - 1] : black[idx];
        float right = (i < W - 1) ? red[idx + 1] : black[idx];
        float bottom = (orig_j < H - 1) ? red[idx + W] : black[idx];
        black[idx] = 0.25f * (top + left + right + bottom - b[idx + offset]);
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

        float top = black[idx];
        float left = black[idx - 1];
        float right = (i < W - 1) ? black[idx + 1] : red[idx];
        float bottom = (orig_j < H - 1) ? black[idx + W] : red[idx];
        red[idx] = (1 - w_opt) * red[idx] + w_opt * 0.25f * (top + left + right + bottom - b[idx]);
    }
    else
    {
        int orig_j = j * 2;
        if (orig_j >= H) return;

        float top = (orig_j > 0) ? black[idx - W] : red[idx];
        float left = (i > 0) ? black[idx - 1] : red[idx];
        float right = (i < W - 1) ? black[idx + 1] : red[idx];
        float bottom = (orig_j < H - 1) ? black[idx] : red[idx];
        red[idx] = (1 - w_opt) * red[idx] + w_opt * 0.25f * (top + left + right + bottom - b[idx]);
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

        float top = (orig_j > 0) ? red[idx - W] : black[idx];
        float left = red[idx - 1];
        float right = (i < W - 1) ? red[idx + 1] : black[idx];
        float bottom = (orig_j < H - 1) ? red[idx] : black[idx];
        black[idx] = (1 - w_opt) * black[idx] + w_opt * 0.25f * (top + left + right + bottom - b[idx + offset]);
    }
    else
    {
        int orig_j = j * 2 + 1;
        if (orig_j >= H) return;

        float top = red[idx];
        float left = (i > 0) ? red[idx - 1] : black[idx];
        float right = (i < W - 1) ? red[idx + 1] : black[idx];
        float bottom = (orig_j < H - 1) ? red[idx + W] : black[idx];
        black[idx] = (1 - w_opt) * black[idx] + w_opt * 0.25f * (top + left + right + bottom - b[idx + offset]);
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
    const int iterations, const int checkFrequency, const float tolerance,
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
    float h_error;
    float *d_error;
    cudaMalloc(&d_error, sizeof(float));

    if (method <= 2) 
    {                
        dim3 nthreadsMethod(16, 16, 1);
        dim3 nblocksMethod((W + nthreadsMethod.x - 1) / nthreadsMethod.x, (H + nthreadsMethod.y - 1) / nthreadsMethod.y, 1);

        float *d_current, *d_result;
        cudaMalloc(&d_current, N * sizeof(float));
        cudaMemset(d_current, 0.0, N * sizeof(float));
        if (d_I_log != nullptr) cudaMemcpy(d_current, d_I_log, N * sizeof(float), cudaMemcpyDeviceToDevice);

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
                cudaMemset(d_error, 0.0, sizeof(float));
                atomicAddBlockErrorsKernel<<<nblocksError, MAX_THREADS>>>(d_current, d_result, d_error, N);
                cudaDeviceSynchronize();
                cudaMemcpy(&h_error, d_error, sizeof(float), cudaMemcpyDeviceToHost);
                h_error /= N;

                if (h_error < tolerance) break;
            }
        }

        if (i % 2 == 1) std::swap(d_current, d_result);
        cudaMemcpy(d_I_log, d_result, N * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaFree(d_current);
        cudaFree(d_result);
        cudaFree(d_error);
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
        if (d_I_log != nullptr) fillInRedBlackInitGaussSeidel2Kernel<<<nblocksFillIn, nthreadsFillIn>>>(H, W, d_I_log, red, black);

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
                cudaMemset(d_error, 0.0, sizeof(float));
                atomicAddBlockErrorsKernel<<<nblocksError, MAX_THREADS>>>(red, prev_red, d_error, N2);
                cudaDeviceSynchronize();
                cudaMemcpy(&h_error, d_error, sizeof(float), cudaMemcpyDeviceToHost);
                h_error /= N2;

                if (h_error < tolerance) break;
            }
        }

        fillInGaussSeidel2Kernel<<<nblocksFillIn, nthreadsFillIn>>>(H, W, red, black, d_I_log);
        cudaDeviceSynchronize();

        cudaFree(red);
        cudaFree(black);
        cudaFree(prev_red);
        cudaFree(red_black_divG);
        cudaFree(d_error);
        return i;
    }

    cudaFree(d_error);
    return -1;
}


__global__ void computeResidualKernel(const int H, const int W, const float* b_h, const float* u_h, float* r_h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= W || j >= H) return;

    int idx = j * W + i;

    float top = (j > 0) ? u_h[idx - W] : u_h[idx];
    float left = (i > 0) ? u_h[idx - 1] : u_h[idx];
    float right = (i < W - 1) ? u_h[idx + 1] : u_h[idx];
    float bottom = (j < H - 1) ? u_h[idx + W] : u_h[idx];

    r_h[idx] = b_h[idx] - (top + left + right + bottom - 4 * u_h[idx]);
}

__global__ void restrict2DKernel(const int H, const int W, const int H2, const int W2, const float* r_h, float* r_2h) {
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    int j2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i2 >= W2 || j2 >= H2) return;

    int i = i2 * 2;
    int j = j2 * 2;

    int idx = j * W + i;
    int idx2 = j2 * W2 + i2;

    float top = (j > 0) ? r_h[idx - W] : r_h[idx];
    float left = (i > 0) ? r_h[idx - 1] : r_h[idx];
    float right = (i < W - 1) ? r_h[idx + 1] : r_h[idx];
    float bottom = (j < H - 1) ? r_h[idx + W] : r_h[idx];

    float topLeft = (j > 0) ? ((i > 0) ? r_h[idx-W-1] : r_h[idx-W]) : ((i > 0) ? r_h[idx-1] : r_h[idx]);
    float topRight = (j > 0) ? ((i < W - 1) ? r_h[idx-W+1] : r_h[idx-W]) : ((i < W - 1) ? r_h[idx+1] : r_h[idx]);
    float bottomLeft = (j < H - 1) ? ((i > 0) ? r_h[idx+W-1] : r_h[idx+W]) : ((i > 0) ? r_h[idx-1] : r_h[idx]);
    float bottomRight = (j < H - 1) ? ((i < W - 1) ? r_h[idx+W+1] : r_h[idx+W]) : ((i < W - 1) ? r_h[idx+1] : r_h[idx]);

    r_2h[idx2] = 0.0625 * (topLeft + topRight + bottomLeft + bottomRight) + 0.125 * (top + bottom + left + right) + 0.25 * r_h[idx];
}

__global__ void interpolate2DKernel(const int H, const int W, const int W2, const float* E_2h, float* E_h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= W || j >= H) return;

    int idx = j * W + i;

    int i2 = i / 2;
    int j2 = j / 2;
    int idx2 = j2 * W2 + i2;
    
    if (j % 2 == 0)
    {
        if (i % 2 == 0)
        {
            E_h[idx] = E_2h[idx2];
        }
        else
        {
            E_h[idx] = 0.5 * (E_2h[idx2] + E_2h[idx2 + 1]);
        }
    }
    else
    {
        if (i % 2 == 0)
        {
            E_h[idx] = 0.5 * (E_2h[idx2] + E_2h[idx2 + W2]);
        }
        else
        {
            E_h[idx] = 0.5 * (E_2h[idx2] + E_2h[idx2 + 1] + E_2h[idx2 + W2] + E_2h[idx2 + W2 + 1]);
        }
    }
}

__global__ void add2DKernel(const int H, const int W, const float* E_h, float* u_h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= W || j >= H) return;

    int idx = j * W + i;

    u_h[idx] += E_h[idx];
}

int vCycleSolver(    
    const int H, const int W, 
    const float* d_divG, const float* args,
    float* d_I_log)
{
    const int N = H * W;
    const int H2 = std::ceil(H / 2.0), W2 = std::ceil(W / 2.0);
    const int N2 = H2 * W2;

    dim3 nthreads_h(16, 16, 1);
    dim3 nblocks_h((W + nthreads_h.x - 1) / nthreads_h.x, (H + nthreads_h.y - 1) / nthreads_h.y, 1);

    dim3 nthreads_2h(16, 16, 1);
    dim3 nblocks_2h((W2 + nthreads_2h.x - 1) / nthreads_2h.x, (H2 + nthreads_2h.y - 1) / nthreads_2h.y, 1);
    
    // Step 1: Iterate on A_h * u = b_h to reach u_h (say 3 Jacobi or Gauss-Seidel steps)
    int pre_smoothing_iter = simpleSolver(H, W, d_divG, args[0], args+6, args[1], args[4], args[5], d_I_log);
    cudaDeviceSynchronize();

    // Step 2: Restrict the residual r_h = b_h − A_h * u_h to the coarse grid by r_{2h} = R_{h}^{2h} * r_h
    float *d_r_h;
    cudaMalloc(&d_r_h, N * sizeof(float));
    computeResidualKernel<<<nblocks_h, nthreads_h>>>(H, W, d_divG, d_I_log, d_r_h);
    cudaDeviceSynchronize();

    float *d_r_2h;
    cudaMalloc(&d_r_2h, N2 * sizeof(float));
    restrict2DKernel<<<nblocks_2h, nthreads_2h>>>(H, W, H2, W2, d_r_h, d_r_2h);
    cudaDeviceSynchronize();

    // Step 3: Solve A_{2h} * E_{2h} = r_{2h} (or come close to E_{2h} by 3 iterations from E = 0)
    float *d_E_2h;
    cudaMalloc(&d_E_2h, N2 * sizeof(float));
    cudaMemset(d_E_2h, 0.0, N2 * sizeof(float));
    int cycle_smoothing_iter;
    if (std::min(H2, W2) <= args[2])
        cycle_smoothing_iter = simpleSolver(H2, W2, d_r_2h, args[0], args+6, args[3], args[4], args[5], d_E_2h);
    else
        cycle_smoothing_iter = vCycleSolver(H2, W2, d_r_2h, args, d_E_2h);
    cudaDeviceSynchronize();

    // Step 4: Interpolate E_{2h} back to E_h = I_{2h}^h * E_{2h}. Add E_h to u_h
    float *d_E_h;
    cudaMalloc(&d_E_h, N * sizeof(float));
    interpolate2DKernel<<<nblocks_h, nthreads_h>>>(H, W, W2, d_E_2h, d_E_h);
    cudaDeviceSynchronize();

    add2DKernel<<<nblocks_h, nthreads_h>>>(H, W, d_E_h, d_I_log);
    cudaDeviceSynchronize();

    // Step 5: Iterate 3 more times on A_h * u = b_h starting from the improved u_h + E_h.
    int post_smoothing_iter = simpleSolver(H, W, d_divG, args[0], args+6, args[1], args[4], args[5], d_I_log);
    cudaDeviceSynchronize();

    cudaFree(d_r_h);
    cudaFree(d_r_2h);
    cudaFree(d_E_2h);
    cudaFree(d_E_h);

    return pre_smoothing_iter + post_smoothing_iter + cycle_smoothing_iter;
}

int wCycleSolver(    
    const int H, const int W, 
    const float* d_divG, const float* args,
    float* d_I_log)
{
const int N = H * W;
    const int H2 = std::ceil(H / 2.0), W2 = std::ceil(W / 2.0);
    const int N2 = H2 * W2;

    dim3 nthreads_h(16, 16, 1);
    dim3 nblocks_h((W + nthreads_h.x - 1) / nthreads_h.x, (H + nthreads_h.y - 1) / nthreads_h.y, 1);

    dim3 nthreads_2h(16, 16, 1);
    dim3 nblocks_2h((W2 + nthreads_2h.x - 1) / nthreads_2h.x, (H2 + nthreads_2h.y - 1) / nthreads_2h.y, 1);
    
    // Step 1: Iterate on A_h * u = b_h to reach u_h (say 3 Jacobi or Gauss-Seidel steps)
    int pre_smoothing_iter = simpleSolver(H, W, d_divG, args[0], args+6, args[1], args[4], args[5], d_I_log);
    cudaDeviceSynchronize();

    // Step 2: Restrict the residual r_h = b_h − A_h * u_h to the coarse grid by r_{2h} = R_{h}^{2h} * r_h
    float *d_r_h;
    cudaMalloc(&d_r_h, N * sizeof(float));
    computeResidualKernel<<<nblocks_h, nthreads_h>>>(H, W, d_divG, d_I_log, d_r_h);
    cudaDeviceSynchronize();

    float *d_r_2h;
    cudaMalloc(&d_r_2h, N2 * sizeof(float));
    restrict2DKernel<<<nblocks_2h, nthreads_2h>>>(H, W, H2, W2, d_r_h, d_r_2h);
    cudaDeviceSynchronize();

    // Step 3: Solve A_{2h} * E_{2h} = r_{2h} (or come close to E_{2h} by 3 iterations from E = 0)
    float *d_E_2h;
    cudaMalloc(&d_E_2h, N2 * sizeof(float));
    cudaMemset(d_E_2h, 0.0, N2 * sizeof(float));
    int cycle_smoothing_iter1;
    if (std::min(H2, W2) <= args[2])
        cycle_smoothing_iter1 = simpleSolver(H2, W2, d_r_2h, args[0], args+6, args[3], args[4], args[5], d_E_2h);
    else
        cycle_smoothing_iter1 = wCycleSolver(H2, W2, d_r_2h, args, d_E_2h);
    cudaDeviceSynchronize();

    // Step 4: Interpolate E_{2h} back to E_h = I_{2h}^h * E_{2h}. Add E_h to u_h
    float *d_E_h;
    cudaMalloc(&d_E_h, N * sizeof(float));
    interpolate2DKernel<<<nblocks_h, nthreads_h>>>(H, W, W2, d_E_2h, d_E_h);
    cudaDeviceSynchronize();

    add2DKernel<<<nblocks_h, nthreads_h>>>(H, W, d_E_h, d_I_log);
    cudaDeviceSynchronize();

    // Step 5: Iterate 3 more times on A_h * u = b_h starting from the improved u_h + E_h.
    int post_smoothing_iter1 = simpleSolver(H, W, d_divG, args[0], args+6, args[1], args[4], args[5], d_I_log);
    cudaDeviceSynchronize();

    // Repeat from Step 2
    computeResidualKernel<<<nblocks_h, nthreads_h>>>(H, W, d_divG, d_I_log, d_r_h);
    cudaDeviceSynchronize();

    restrict2DKernel<<<nblocks_2h, nthreads_2h>>>(H, W, H2, W2, d_r_h, d_r_2h);
    cudaDeviceSynchronize();

    // Step 3
    cudaMemset(d_E_2h, 0.0, N2 * sizeof(float));
    int cycle_smoothing_iter2;
    if (std::min(H2, W2) <= args[2])
        cycle_smoothing_iter2 = simpleSolver(H2, W2, d_r_2h, args[0], args+6, args[3], args[4], args[5], d_E_2h);
    else
        cycle_smoothing_iter2 = wCycleSolver(H2, W2, d_r_2h, args, d_E_2h);
    cudaDeviceSynchronize();

    // Step 4
    interpolate2DKernel<<<nblocks_h, nthreads_h>>>(H, W, W2, d_E_2h, d_E_h);
    cudaDeviceSynchronize();

    add2DKernel<<<nblocks_h, nthreads_h>>>(H, W, d_E_h, d_I_log);
    cudaDeviceSynchronize();

    // Step 5
    int post_smoothing_iter2 = simpleSolver(H, W, d_divG, args[0], args+6, args[1], args[4], args[5], d_I_log);
    cudaDeviceSynchronize();

    cudaFree(d_r_h);
    cudaFree(d_r_2h);
    cudaFree(d_E_2h);
    cudaFree(d_E_h);

    return pre_smoothing_iter + post_smoothing_iter1 + post_smoothing_iter2 + cycle_smoothing_iter1 + cycle_smoothing_iter2;
}

int fCycleSolver(    
    const int H, const int W, 
    const float* d_divG, const float* args,
    float* d_I_log)
{
    return -1;
}


int multigridSolver(
    const int H, const int W, 
    const float* d_divG, const int method, const float* args,
    const int iterations, const int checkFrequency, const float tolerance,
    float* d_I_log)
{
    const int N = H * W;

    mgMethodFunction mgMethodKernel;
    switch (method) {
        case 5:
            mgMethodKernel = vCycleSolver;
            break;
        case 6:
            mgMethodKernel = wCycleSolver;
            break;
        case 7:
            mgMethodKernel = fCycleSolver;
            break;
        default:
            return -1;
    }

    int nblocksError = (N + MAX_THREADS - 1) / MAX_THREADS;
    float h_error;
    float *d_error;
    cudaMalloc(&d_error, sizeof(float));

    float *d_prev, *d_result;
    cudaMalloc(&d_prev, N * sizeof(float));
    cudaMemset(d_prev, 0.0, N * sizeof(float));
    cudaMalloc(&d_result, N * sizeof(float));
    cudaMemset(d_result, 0.0, N * sizeof(float));
    
    int total_iter_until_convergence = 0;
    for (int i = 0; i < iterations; ) 
    {
        if ((i+1) % checkFrequency == 0) cudaMemcpy(d_prev, d_result, N * sizeof(float), cudaMemcpyDeviceToDevice);
        total_iter_until_convergence += mgMethodKernel(H, W, d_divG, args, d_result);
        cudaDeviceSynchronize(); 
        ++i;

        if (i % checkFrequency == 0)
        {   
            cudaMemset(d_error, 0.0, sizeof(float));
            atomicAddBlockErrorsKernel<<<nblocksError, MAX_THREADS>>>(d_result, d_prev, d_error, N);
            cudaDeviceSynchronize();
            cudaMemcpy(&h_error, d_error, sizeof(float), cudaMemcpyDeviceToHost);
            h_error /= N;

            if (h_error < tolerance) break;
        }
    }

    cudaMemcpy(d_I_log, d_result, N * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(d_result);
    cudaFree(d_error);
    return total_iter_until_convergence;
}