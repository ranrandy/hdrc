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
    
    // Convert color from RGB space to log of luminance, because HDR
    torch::Tensor luminanceCoeffs = torch::tensor({0.2126729, 0.7151522, 0.0721750}, float_opts).view({3, 1, 1});
    torch::Tensor hdr_lum = torch::sum(hdr_rad_map_rgb * luminanceCoeffs, 0);
    hdr_lum = torch::where(hdr_lum == 0.0, torch::tensor(1e-6), hdr_lum);
    torch::Tensor hdr_log_lum = torch::log(hdr_lum);

    // std::cout << "Mean: " << torch::mean(hdr_lum).item<float>() << ", " << torch::mean(hdr_log_lum).item<float>() << std::endl;
    // std::cout << "Min: " << torch::min(hdr_lum).item<float>() << ", " << torch::min(hdr_log_lum).item<float>() << std::endl;
    // std::cout << "Max: " << torch::max(hdr_lum).item<float>() << ", " << torch::max(hdr_log_lum).item<float>() << std::endl;

    // Create the gaussian pyramid. Finest at the bottom[0]. Coarsest at the top[L-1].
    std::vector<torch::Tensor> pyramid = buildGaussianPyramid(hdr_log_lum, L, H, W);

    // Calculate the scaling factor at each of level of the pyramid
    std::vector<torch::Tensor> scaling_factor_pyramid;
    for (int level = 0; level < pyramid.size(); ++level) {
        scaling_factor_pyramid.push_back(calculateScalings(pyramid[level], level, alpha, beta));
    }

    // Calculate the attenuation at the finest level (starting from the coarsest level).
    torch::Tensor attenuation = scaling_factor_pyramid.back();
    for (int level = scaling_factor_pyramid.size() - 2; level >= 0; --level) {
        auto target_size = scaling_factor_pyramid[level].sizes();
        auto resized_attenuation = torch::nn::functional::interpolate(
            attenuation.unsqueeze(0).unsqueeze(0), 
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{target_size[0], target_size[1]})
                .mode(torch::kBilinear)
                .align_corners(false)
            ).squeeze(0).squeeze(0);
        attenuation = resized_attenuation * scaling_factor_pyramid[level];
    }

    // Calculate attenuated gradients, namely G(x, y)
    torch::Tensor d_div_G = calculateAttenuatedDivergence(hdr_log_lum, attenuation).to(torch::kCUDA);

    // Solve the poisson equation
    torch::Tensor h_I_log = torch::full({H, W}, 0.0, float_opts);

    float* arguments;

    float *d_I_log;
    cudaMalloc(&d_I_log, H * W * sizeof(float));
    cudaMemset(d_I_log, 0.0, H * W * sizeof(float));

    int iter_converge = 0;

    cudaMallocHost(&arguments, 7 * sizeof(float));
    arguments[0] = 3;
    arguments[1] = 20;
    arguments[2] = 100;
    arguments[3] = 1000;
    arguments[4] = 5;
    arguments[5] = 0.0001;
    arguments[6] = 1.45;

    iter_converge = multigridSolver(
        H, W, d_div_G.contiguous().data<float>(),
        5, arguments,
        1, 10, 0.0001,
        d_I_log);
    cudaDeviceSynchronize();

    cudaMemcpy(h_I_log.contiguous().data<float>(), d_I_log, H * W * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_I_log);
    cudaFreeHost(arguments);

    // std::cout << "h_I_log: " 
    //           << torch::mean(h_I_log).item<float>() << ", " 
    //           << torch::min(h_I_log).item<float>() << ", " 
    //           << torch::max(h_I_log).item<float>() << std::endl;

    torch::Tensor I = normalize(torch::exp(h_I_log));

    // std::cout << "I: " 
    //           << torch::mean(I).item<float>() << ", " 
    //           << torch::min(I).item<float>() << ", " 
    //           << torch::max(I).item<float>() << std::endl;

    // Generate the LDR output
    torch::Tensor ldr_out_color = torch::pow(torch::div(hdr_rad_map_rgb, hdr_lum.unsqueeze(0)), saturation) * I;

    // std::cout << "ldr_out_color: " 
    //           << torch::mean(ldr_out_color).item<float>() << ", " 
    //           << torch::min(ldr_out_color).item<float>() << ", " 
    //           << torch::max(ldr_out_color).item<float>() << std::endl;

    return (torch::clamp(ldr_out_color, 0.0, 1.0) * 255).to(torch::kUInt8);
}