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
    for (int j = 0; j < _H; ++j) for (int i = 0; i < _W; ++i) _real_f_laplacian[j * _W + i] = getRealLaplacian(j, i);

    // Initialize the estimated f laplacian array
    _estimated_f_laplacian = new float [_H * _W];
    for (int j = 0; j < _H; ++j) for (int i = 0; i < _W; ++i) _estimated_f_laplacian[j * _W + i] = getEstimatedLaplacian(j, i);
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
    int idx = j * _W + i;

    // Simulate Neumann Boundary condition
    float up = (j > 0) ? _f[idx - _W] : _f[idx];
    float left = (i > 0) ? _f[idx - 1] : _f[idx];
    float right = (i < _W - 1) ? _f[idx + 1] : _f[idx];
    float bottom = (j < _H - 1) ? _f[idx + _W] : _f[idx];

    return up + left + right + bottom - 4*_f[idx];
}
