#include <torch/extension.h>
#include <vector>
#include <tuple>
#include "solvers.h"

int solve(
    const int H, const int W,
    const torch::Tensor& d_div_G, 
    const int method, const std::vector<float>& args,
    const int iterations, const int checkFrequency, const float tolerance,
    torch::Tensor& h_I_log);