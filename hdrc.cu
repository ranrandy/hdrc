#include "hdrc.h"

torch::Tensor hdrcCUDA(
    const torch::Tensor& hdr_rad_map_rgb, 
    const float alpha, 
    const float beta, 
    const float saturation)
{
    const int H = hdr_rad_map_rgb.size(-2);
    const int W = hdr_rad_map_rgb.size(-1);

    torch::Tensor out_log_lum = torch::full({H, W}, 0.0, torch::kFloat32);
    
    // Convert color from RGB space to log of luminance, because HDR
    torch::Tensor luminanceCoeffs = torch::tensor({0.2126729, 0.7151522, 0.0721750},
                                                  torch::kFloat32).view({3, 1, 1});
    torch::Tensor hdr_lum = torch::sum(hdr_rad_map_rgb * luminanceCoeffs, 0);
    torch::Tensor hdr_log_lum = torch::log(hdr_lum);

    // Apply gradient domain HDR compression. Line 189-215 in hdrc.py.
    HDRC::DynamicRangeCompressor::compress(alpha, beta, H, W,
        hdr_log_lum.contiguous().data_ptr<float>(),
        out_log_lum.contiguous().data_ptr<float>()
    );

    // Generate the LDR output
    torch::Tensor ldr_out_color = torch::pow(torch::div(hdr_rad_map_rgb, hdr_lum.unsqueeze(-1)), saturation) * torch::exp(out_log_lum);
    return ldr_out_color;
}