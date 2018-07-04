#ifndef CONVOLUTION_H
#define CONVOLUTION_H
#include <cmath>
#include "layer.h"

class Convolution: public Layer {
public:
    Convolution(int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h,
                int stride_w, int input_channels, int output_channels, const string& layer_name,
                float mean = 0.0, float stddev = 0.1, float bias_value = 0.1);
    virtual void set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks);
    virtual void forward(const vector<Chunk*>& input, const vector<Chunk*>& output);
    virtual void backward(const vector<Chunk*>& input, const vector<Chunk*>& output);
private:
    void initialize(float mean, float stddev, float bias_value);
    int kernel_h_, kernel_w_;
    int pad_h_, pad_w_;
    int stride_h_, stride_w_;
    int input_channels_;
    int output_channels_;
};

#endif // CONVOLUTION_H
