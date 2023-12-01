#include "hdrc.h"

torch::Tensor hdrcCUDA(
    const torch::Tensor& hdr_rad_map_rgb, 
    const float alpha, 
    const float beta, 
    const float saturation)
{
    const int H = hdr_rad_map_rgb.size(-2);
    const int W = hdr_rad_map_rgb.size(-1);
    const int L = int(std::min(std::log(H / 32), std::log(W / 32))) + 1;

    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32);

    torch::Tensor out_log_lum = torch::full({H, W}, 0.0, torch::kFloat32);
    
    // Convert color from RGB space to log of luminance, because HDR
    torch::Tensor luminanceCoeffs = torch::tensor({0.2126729, 0.7151522, 0.0721750}, float_opts).view({3, 1, 1});
    torch::Tensor hdr_lum = torch::sum(hdr_rad_map_rgb * luminanceCoeffs, 0);
    torch::Tensor hdr_log_lum = torch::log(hdr_lum);

    // Create the gaussian pyramid. Finest at the bottom[0]. Coarsest at the top[L-1].
    std::vector<at::Tensor> pyramid = buildGaussianPyramid(hdr_log_lum, H, W, L);

    // Calculate the scaling factor at each of level of the pyramid

    // Calculate the attenuation at the finest level (starting from the coarsest level).


    // // # Calculate attenuated gradients, namely G(x, y)
    // // attenuated_grad_x, attenuated_grad_y = calculate_attenuated_gradients(hdr_lum_log, attenuation)

    // // # Calculate \Div{G(x, y)}
    // // div_G = calculate_divergence(attenuated_grad_x, attenuated_grad_y)

    // // # Solve the Poisson linear equations to find I(x, y) from G(x, y) using the Jacobi iteration method
    // // I_log = solve_poisson_equation(div_G, args)
    // // Apply gradient domain HDR compression. Line 189-215 in hdrc.py.

    // HDRC::DynamicRangeCompressor::compress(alpha, beta, H, W,
    //     hdr_log_lum.contiguous().data_ptr<float>(),
    //     out_log_lum.contiguous().data_ptr<float>()
    // );

    // Generate the LDR output
    torch::Tensor ldr_out_color = torch::pow(torch::div(hdr_rad_map_rgb, hdr_lum.unsqueeze(-1)), saturation) * torch::exp(out_log_lum);
    return ldr_out_color;
}