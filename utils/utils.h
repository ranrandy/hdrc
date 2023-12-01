#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <torch/torch.h>
#include <cmath>
#include <vector>


// Build the gaussian pyramid
at::Tensor createGaussianKernel(int kernel_size, double std);

at::Tensor gaussianBlur(const at::Tensor& input, int H, int W, int kernel_size, double std);

std::vector<at::Tensor> buildGaussianPyramid(at::Tensor image, int H, int W, int levels);



#endif