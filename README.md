# HDRC
Implementation of Gradient Domain HDR Compression in CUDA.

Gamma vs Gradient Domain Compression

<img src="output/bigFogMap_ldr_gamma.png" width="400"/>  <img src="output/bigFogMap_ldr_norm.png" width="400"/>

Attenuation map of the Belgium House scene.

<img src="./output/belgium_attenuation.png" alt="belgium_attenuation" width="400">

### References
Fattal, R., Lischinski, D., & Werman, M. (2023). _Gradient domain high dynamic range compression._ In Seminal Graphics Papers: Pushing the Boundaries, Volume 2 (pp. 671-678).

### Problems
1. The poisson solver, if using the residual to stop iterating, can only deal with images with fewer than 2^30 pixels for now, and the implementation for calculating residual/error between previous and the current results may be efficient.