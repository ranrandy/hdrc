#include "utils.h"


torch::Tensor createGaussianKernel(const int kernel_size, const float sigma) {
    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor kernel = torch::zeros({kernel_size, kernel_size}, float_opts);
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

torch::Tensor gaussianBlur(const torch::Tensor& input, const int H, const int W, const int kernel_size, const float sigma) {
    torch::Tensor kernel = createGaussianKernel(kernel_size, sigma);
    int padding_size = static_cast<int>(kernel_size / 2);

    return torch::nn::functional::conv2d(
        input.view({1, 1, H, W}), kernel, 
        torch::nn::functional::Conv2dFuncOptions().padding(padding_size)).view({H, W});
}

std::vector<torch::Tensor> buildGaussianPyramid(const torch::Tensor& image, const int levels, const int H, const int W) {
    std::vector<torch::Tensor> pyramid;
    pyramid.push_back(image);

    int current_H = H, current_W = W;
    for (size_t i = 1; i < levels; ++i) {
        torch::Tensor blurred = gaussianBlur(pyramid[i - 1], current_H, current_W, 5, 1.0);

        current_H /= 2;
        current_W /= 2;
        torch::Tensor downsampled = torch::nn::functional::interpolate(
            blurred.unsqueeze(0).unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{current_H, current_W})
                .mode(torch::kBilinear)
                .align_corners(false)
            ).squeeze(0).squeeze(0);
        pyramid.push_back(downsampled);
    }
    return pyramid;
}

torch::Tensor calculateScalings(const torch::Tensor& pyramid_i, const int level, const float alpha, const float beta) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto grad_filter_x = torch::tensor({{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}}, options).view({1, 1, 3, 3});
    auto grad_filter_y = torch::tensor({{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}}, options).view({1, 1, 3, 3});

    // Caculate gradient magnitude at this level
    auto grad_x = torch::nn::functional::conv2d(
        pyramid_i.unsqueeze(0).unsqueeze(0), grad_filter_x, 
        torch::nn::functional::Conv2dFuncOptions().padding(1)).squeeze();
    auto grad_y = torch::nn::functional::conv2d(
        pyramid_i.unsqueeze(0).unsqueeze(0), grad_filter_y, 
        torch::nn::functional::Conv2dFuncOptions().padding(1)).squeeze();
    float scale = std::pow(2, level + 1);
    grad_x /= scale;
    grad_y /= scale;

    // Calculate gradient magnitude
    torch::Tensor grad_mag = torch::sqrt(torch::pow(grad_x, 2) + torch::pow(grad_y, 2));
    grad_mag = torch::where(grad_mag == 0.0, torch::tensor(1e-6), grad_mag);

    // Determine the gradient scaling factor by gradient magnitude
    auto alpha_scaled = alpha * torch::mean(grad_mag);
    return (alpha_scaled / grad_mag) * torch::pow((grad_mag / alpha_scaled), beta);
}

torch::Tensor calculateAttenuatedDivergence(torch::Tensor lum_log, torch::Tensor phi) {
    auto Gx = torch::zeros_like(lum_log, torch::kFloat32);
    auto Gy = torch::zeros_like(lum_log, torch::kFloat32);

    Gx.slice(1, 0, -1) = (lum_log.slice(1, 1, lum_log.size(1)) - lum_log.slice(1, 0, -1)) * phi.slice(1, 0, -1);
    Gy.slice(0, 0, -1) = (lum_log.slice(0, 1, lum_log.size(0)) - lum_log.slice(0, 0, -1)) * phi.slice(0, 0, -1);

    auto div_G = torch::zeros_like(Gx, torch::kFloat32);

    div_G += Gx;
    div_G += Gy;

    div_G.slice(1, 1, Gx.size(1)) -= Gx.slice(1, 0, -1);
    div_G.slice(0, 1, Gx.size(0)) -= Gy.slice(0, 0, -1);

    return div_G;
}

torch::Tensor normalize(torch::Tensor input) {
    auto min_val = input.min();
    auto max_val = input.max();
    return (input - min_val) / (max_val - min_val);
}