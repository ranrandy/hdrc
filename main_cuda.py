'''.
Python wrapper to call the CUDA Gradient Domain HDR Compressor.

Current performance: ~150 milliseconds for the belgium scene. Unfortunately, this is still not that good enough be considered as real-time.

Reference: https://github.com/Ockhius/hdr_tonemapping_fattal02
'''

import torch
import cv2
import argparse
from hdrc._C import hdrcCUDA
import time

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Gradient Domain HDR Radiance Map Tone Mapping")

    arg_parser.add_argument("--source", type=str, default="data/belgium.hdr", help="Source HDR radiance map path")
    arg_parser.add_argument("--output_folder", type=str, default="output", help="Output LDR image folder")

    arg_parser.add_argument("--alpha", type=float, default=0.10, help="Max \"small\" gradient")
    arg_parser.add_argument("--beta", type=float, default=0.90, help="Attenuation factor for large gradients")
    arg_parser.add_argument("--saturation", type=float, default=0.45, help="Color saturation of the resulting image")

    args = arg_parser.parse_args()

    # Read HDR radiance map
    hdr_rad_map_rgb = cv2.imread(args.source, cv2.IMREAD_UNCHANGED)[:, :, ::-1]

    flipped_hdr_rad_map_rgb = torch.tensor(hdr_rad_map_rgb.transpose(2, 0, 1).copy()).to(torch.float32)
    
    # Start Timer
    start_time = time.time()

    output = hdrcCUDA(flipped_hdr_rad_map_rgb, args.alpha, args.beta, args.saturation)
    output = output.numpy().transpose(1, 2, 0)[:, :, ::-1]
    
    # End Timer
    end_time = time.time()
    duration = (end_time - start_time) * 1000 # Milliseconds
    print(f"Duration: {duration} milliseconds")
    
    # save the output
    cv2.imwrite(f"{args.output_folder}/all_cuda_{args.source.split('/')[-1][:-4]}_ldr.png", output)

    print("Done.")