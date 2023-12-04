#include <math.h>
#include <cmath>
#include <math.h>
#include <iomanip>
#include <iostream>

#include <omp.h>
#include <stdio.h>

class Function2D {
private:
    int _W = 1600;
    int _H = 1200;
public:
    Function2D(int W, int H);
    ~Function2D();

    float* _f;
    float* _real_f_laplacian;
    float* _estimated_f_laplacian;

    float f(int j, int i);
    float getRealLaplacian(int j, int i);
    float getEstimatedLaplacian(int j, int i);
};