'''
Python Implementation of the "Gradient Domain High Dynamic Range Compression" paper by Raanan Fattal in 2002.

Expected 4.5 seconds for 1024 x 768 HDR radiance map on a 1800 MHz Pentium 4.

We will use this python code to guide our CUDA implementation and hopefully achieve real-time (30-60 fps) performance
to tone map a raw image from 3D HDR Gaussians Splatting on a single 4060Ti GPU.

Reference: https://github.com/Ockhius/hdr_tonemapping_fattal02
'''

import numpy as np
import cv2
import argparse
from skimage import filters

def build_gaussian_pyramid(img):
    '''
    Build the Gaussian pyramid.
    '''
    img_pyramid = [img]
    while (img_pyramid[-1].shape[0] / 2 > 32 and img_pyramid[-1].shape[1] / 2 > 32):
        img_blurred = filters.gaussian(img_pyramid[-1])
        img_pyramid.append(img_blurred[::2, ::2])
    return img_pyramid

def calculate_pyramid_gradient_magnitudes(pyramid_i, level):
    '''
    Calculate the gradient magnitude at each level of the Gaussian pyramid.
    '''
    grad_x = np.zeros(pyramid_i.shape).astype(np.float32)
    grad_y = np.zeros(pyramid_i.shape).astype(np.float32)
    for j in range(pyramid_i.shape[0]):
        for i in range(pyramid_i.shape[1]):
            # Column boundary conditions 
            if j == 0:
                grad_y[j, i] = (pyramid_i[j + 1, i] - pyramid_i[j, i]) / (2 ** level)
            elif j == pyramid_i.shape[0] - 1:
                grad_y[j, i] = (pyramid_i[j, i] - pyramid_i[j - 1, i]) / (2 ** level)
            else:
                grad_y[j, i] = (pyramid_i[j + 1, i] - pyramid_i[j - 1, i]) / (2 ** (level + 1))
            # Row boundary condition
            if i == 0:
                grad_x[j, i] = (pyramid_i[j, i + 1] - pyramid_i[j, i]) / (2 ** level)
            elif i == pyramid_i.shape[1] - 1:
                grad_x[j, i] = (pyramid_i[j, i] - pyramid_i[j, i - 1]) / (2 ** level)
            else:
                grad_x[j, i] = (pyramid_i[j, i + 1] - pyramid_i[j, i - 1]) / (2 ** (level + 1))
    return np.sqrt(grad_x ** 2 + grad_y ** 2)

def calculate_scalings(grad_mag, args):
    '''
    Calculate the scaling factor at each level of the pyramid by
        scaling = (alpha / grad_mag) * (grad_mag / alpha) ^ beta
    '''
    alpha = args.alpha * np.mean(grad_mag)
    return (alpha / grad_mag) * (grad_mag / alpha) ** args.beta

def resize(img, new_shape):
    '''
    Resize the scaling factors to (H*2, W*2) with bilinear interpolation.
    '''
    H, W = img.shape
    H2, W2 = new_shape
    resized_img = np.zeros((H2, W2), dtype=img.dtype)
    
    for j in range(H2):
        for i in range(W2):
            y, x = j / 2, i / 2 
            y_low, x_low = int(np.floor(y)), int(np.floor(x))
            y_high, x_high = min(y_low + 1, H - 1), min(x_low + 1, W - 1)
            dx, dy = x - x_low, y - y_low

            # Bilinear interpolation
            resized_img[j, i] = (1 - dx) * (1 - dy) * img[y_low, x_low] + \
                                dx * (1 - dy) * img[y_low, x_high] + \
                                (1 - dx) * dy * img[y_high, x_low] + \
                                dx * dy * img[y_high, x_high]
    return resized_img

def calculate_attenuated_gradients(lum_log, phi):
    '''
    Calculate G(x, y) = \grad{H} * phi(x, y)
    '''
    attenuated_grad_x = np.zeros_like(lum_log, dtype=np.float32)
    attenuated_grad_y = np.zeros_like(lum_log, dtype=np.float32)
    for j in range(lum_log.shape[0]):
        for i in range(lum_log.shape[1]):
            if i + 1 >= lum_log.shape[1]:
                attenuated_grad_x[j, i] = (lum_log[j, lum_log.shape[1] - 2] - lum_log[j, i]) * 0.5 * (phi[j, lum_log.shape[1] - 2] + phi[j, i])
            else:
                attenuated_grad_x[j, i] = (lum_log[j, i + 1] - lum_log[j, i]) * 0.5 * (phi[j, i + 1] + phi[j, i])
            if j + 1 >= lum_log.shape[0]:
                attenuated_grad_y[j, i] = (lum_log[lum_log.shape[0] - 2, i] - lum_log[j, i]) * 0.5 * (phi[lum_log.shape[0] - 2, i] + phi[j, i])
            else:
                attenuated_grad_y[j, i] = (lum_log[j + 1, i] - lum_log[j, i]) * 0.5 * (phi[j + 1, i] + phi[j, i])
    return attenuated_grad_x, attenuated_grad_y

def calculate_divergence(Gx, Gy):
    '''
    Calculate divergence of the vector field G.
    '''
    divG = np.zeros_like(Gx, dtype=np.float32)
    for j in range(Gx.shape[0]):
        for i in range(Gx.shape[1]):
            divG[j, i] = Gx[j, i] + Gy[j, i]
            if i > 0:
                divG[j, i] -= Gx[j, i - 1]
            if j > 0:
                divG[j, i] -= Gy[j - 1, i]
            if i == 0:
                divG[j, i] += Gx[j, i]
            if j == 0:
                divG[j, i] += Gy[j, i]
    return divG

def solve_poisson_equation(div_G, args):
    '''
    Apply the Jacobi iteration method to solve the Poisson equation and get I(x, y) from G(x, y)
    '''
    I = np.zeros_like(div_G, dtype=np.float32)
    for _ in range(args.max_iterations):
        prev_I = I.copy()
        for i in range(1, I.shape[0] - 1):
            for j in range(1, I.shape[1] - 1):
                I[i, j] = 0.25 * (prev_I[i+1, j] + prev_I[i-1, j] + prev_I[i, j+1] + prev_I[i, j-1] - div_G[i, j])
        if np.linalg.norm(I - prev_I) < args.tolerance:
            break
    return I

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Gradient Domain HDR Radiance Map Tone Mapping")
    arg_parser.add_argument("--source", type=str, default="data/belgium.hdr", help="Source HDR radiance map path")
    arg_parser.add_argument("--output_folder", type=str, default="output", help="Output LDR image folder")
    arg_parser.add_argument("--save_att", action="store_true", help="Save gradient attenuation map")
    arg_parser.add_argument("--alpha", type=float, default=0.18, help="Max \"small\" gradient")
    arg_parser.add_argument("--beta", type=float, default=0.87, help="Attenuation factor for large gradients")
    arg_parser.add_argument("--saturation", type=float, default=0.55, help="Color saturation of the resulting image")
    arg_parser.add_argument("--max_iterations", type=int, default=1000, help="Max number of iterations to run the Jacobi Poisson solver")
    arg_parser.add_argument("--tolerance", type=float, default=1e-6, help="Tolerance to stop iterating the Jacobi Poisson solver")
    args = arg_parser.parse_args()

    # Read HDR radiance map
    hdr_rad_map_rgb = cv2.imread(args.source, cv2.IMREAD_UNCHANGED)[:, :, ::-1]

    # Covert RGB color to XYZ space
    hdr_rad_map_xyz = cv2.cvtColor(hdr_rad_map_rgb, cv2.COLOR_RGB2XYZ)

    # We treat HDR maps as (scalar) luminance (Y) functions
    hdr_lum_log = np.log(hdr_rad_map_xyz[:, :, 1])

    # Get gaussian pyramid. Finest at the bottom[0]. Coarsest on the top[L-1].
    pyramid = build_gaussian_pyramid(hdr_lum_log)

    # Calculate the scaling factor at each of level of the pyramid
    scaling_factor_pyramid = []
    for level in range(len(pyramid)):
        # Caculate gradient magnitude at each level
        grad_mag = calculate_pyramid_gradient_magnitudes(pyramid[level], level)
        grad_mag[grad_mag == 0.0] = 1e-6
        # Determine the gradient scaling factor by gradient magnitude
        scaling_factor_pyramid.append(calculate_scalings(grad_mag, args))

    # Calculate the attenuation at the finest level (starting from the coarsest level).
    attenuation = scaling_factor_pyramid[-1]
    for level in range(len(scaling_factor_pyramid)-2, -1, -1):
        attenuation = resize(attenuation, scaling_factor_pyramid[level].shape) * scaling_factor_pyramid[level]
    if args.save_att:
        cv2.imwrite(f"{args.output_folder}/{args.source.split('/')[-1][:-4]}_attenuation.png", (np.clip(attenuation, 0.0, 1.0) * 255.0).astype(np.uint8))

    # Calculate attenuated gradients, namely G(x, y)
    attenuated_grad_x, attenuated_grad_y = calculate_attenuated_gradients(hdr_lum_log, attenuation)

    # Calculate \Div{G(x, y)}
    div_G = calculate_divergence(attenuated_grad_x, attenuated_grad_y)

    # Solve the Poisson linear equations to find I(x, y) from G(x, y) using the Jacobi iteration method
    I_log = solve_poisson_equation(div_G, args)
    I = np.exp(I_log)

    # Produce the LDR output
    output = np.zeros(hdr_lum_log.shape).astype(np.float32)
    for c in range(3):
        output[:, :, c] = (hdr_rad_map_rgb[:, :, c] / hdr_rad_map_xyz[:, :, 1]) ** args.saturation * I
    
    # Rescale the output
    output = (np.clip(output, 0.0, 1.0) * 255.0).astype(np.uint8)[:, :, ::-1]

    # Save the output
    cv2.imwrite(f"{args.output_folder}/{args.source.split('/')[-1][:-4]}_ldr.png", output)

    print("Done.")