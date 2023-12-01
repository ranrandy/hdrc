#ifndef COMPRESSOR_H_INCLUDED
#define COMPRESSOR_H_INCLUDED

namespace HDRC
{
    class DynamicRangeCompressor
    {
    public:
        static void compress(
            const float alpha,
            const float beta,
            const int H, 
            const int W,
            const float* hdr_log_lum,
            float* out_log_lum
        );
    };
};

#endif