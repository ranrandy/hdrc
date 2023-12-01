#include "utils.h"


at::Tensor createGaussianKernel(int kernel_size, double sigma) {
    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32);
    at::Tensor kernel = torch::zeros({kernel_size, kernel_size}, float_opts);
    int center = static_cast<int>(kernel_size / 2);

    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            float x = static_cast<float>(i - center);
            float y = static_cast<float>(j - center);
            kernel[i][j] = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
        }
    }
    kernel /= kernel.sum();
    return kernel.view({1, 1, kernel_size, kernel_size});
}


at::Tensor gaussianBlur(const at::Tensor& input, int H, int W, int kernel_size, double sigma) {
    at::Tensor kernel = createGaussianKernel(kernel_size, sigma);
    int padding_size = static_cast<int>(kernel_size / 2);

    return torch::nn::functional::conv2d(
        input.view({1, 1, H, W}), kernel, 
        torch::nn::functional::Conv2dFuncOptions().padding(padding_size)).view({H, W});
}


std::vector<at::Tensor> buildGaussianPyramid(at::Tensor image, int levels, int H, int W) {
    std::vector<at::Tensor> pyramid;
    pyramid.push_back(image);

    for (size_t i = 1; i < levels; ++i) {
        at::Tensor blurred = gaussianBlur(pyramid[i - 1], H, W, 5, 1.0);

        int new_H = static_cast<int>(H / 2);
        int new_W = static_cast<int>(W / 2);
        at::Tensor downsampled = torch::nn::functional::interpolate(
            blurred,
            torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({new_H, new_W})).mode(torch::kNearest)
            );
        pyramid.push_back(downsampled);
    }
    return pyramid;
}