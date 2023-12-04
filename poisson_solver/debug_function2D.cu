#define _USE_MATH_DEFINES

#include <cmath>
#include "debug_function2D.h"

Function2D::Function2D(int W, int H) {
    _W = W;
    _H = H;

    // Store the function values
    _f = new float [_H * _W];
    for (int j = 0; j < _H; ++j) for (int i = 0; i < _W; ++i) _f[j * _W + i] = f(j, i); 

    // Store the real laplacian values
    _real_f_laplacian = new float [_H * _W];
    for (int j = 1; j < _H-1; ++j) for (int i = 1; i < _W-1; ++i) _real_f_laplacian[j * _W + i] = getRealLaplacian(j, i);

    // Initialize the estimated f laplacian array
    _estimated_f_laplacian = new float [_H * _W];
    for (int j = 1; j < _H-1; ++j) for (int i = 1; i < _W-1; ++i) _estimated_f_laplacian[j * _W + i] = getEstimatedLaplacian(j, i);
}

Function2D::~Function2D() {
    delete[] _f;
    delete[] _real_f_laplacian;
    delete[] _estimated_f_laplacian;
}

float Function2D::f(int j, int i) {
    return sin(M_PI * (i + j) / 100.0);
}

float Function2D::getRealLaplacian(int j, int i) {
    return (-1) * M_PI * M_PI / 5000.0 * sin(M_PI * (i + j) / 100.0);
}

float Function2D::getEstimatedLaplacian(int j, int i) {
    return _f[j * _W + i-1] + _f[j * _W + i+1] + _f[(j-1) * _W + i] + _f[(j+1) * _W + i] - 4*_f[j * _W + i];
}
