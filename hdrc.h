#pragma once
#include <torch/extension.h>
#include "utils/compressor.h"

/*
hdr_rad_map:
    (NUM_CHANNELS=3, Height, Width) torch tensor.
    HDR radiance map output from the RawGS.
    Ideally noise-free.
    Already demoisacked.
    Already on GPU.
*/
torch::Tensor hdrcCUDA(
    const torch::Tensor& hdr_rad_map, 
    const float alpha = 0.18, 
    const float beta = 0.87, 
    const float saturation = 0.55
    );