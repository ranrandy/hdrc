#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <torch/torch.h>
#include <cmath>
#include <vector>

// Build the gaussian pyramid
torch::Tensor createGaussianKernel(const int kernel_size, const float sigma);

torch::Tensor gaussianBlur(const torch::Tensor& input, const int H, const int W, const int kernel_size, const float sigma);

std::vector<torch::Tensor> buildGaussianPyramid(const torch::Tensor& image, const int levels, const int H, const int W);

// Calculate scalings
torch::Tensor calculateScalings(const torch::Tensor& pyramid_i, const int level, const float alpha, const float beta);

// Calculate attenuated divergence
torch::Tensor calculateAttenuatedDivergence(torch::Tensor lum_log, torch::Tensor phi);

// Min-max
torch::Tensor normalize(torch::Tensor input);

#endif