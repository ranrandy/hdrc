#pragma once
#include <algorithm>
#include <torch/extension.h>
#include <tuple>
#include "utils\\utils.h"
#include "poisson_solvers\\solvers.h"

/*
hdr_rad_map:
    (NUM_CHANNELS=3, Height, Width) torch tensor.
    HDR radiance map output from the RawGS.
    Ideally noise-free.
    Already demoisacked.
    Already on GPU. (This has not been integrated)
*/
torch::Tensor hdrcCUDA(
    const torch::Tensor& hdr_rad_map, 
    const float alpha = 0.18, 
    const float beta = 0.87, 
    const float saturation = 0.55
    );