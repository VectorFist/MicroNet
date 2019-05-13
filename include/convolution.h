#ifndef CONVOLUTION_H
#define CONVOLUTION_H
#include <cmath>
#include <memory>
#include "layer.h"

namespace micronet {

class Convolution: public Layer {
public:
    Convolution(): col_tmp_(new Chunk), all_one_tmp_(new Chunk) {};
    Convolution(int kernel_h, int kernel_w, int stride_h,
                int stride_w, int output_channels, const string& padding = "valid",
                float mean = 0.0, float stddev = 0.1, float bias_value = 0.1,
                const string& layer_name = "convolution");
    virtual void forward(bool is_train=true) override;
    virtual void backward() override;
    chunk_ptr operator()(const chunk_ptr& in_chunk);

protected:
    virtual vector<int> shape_inference() override;

private:
    void initialize();
    void pad_inference();
    chunk_ptr col_tmp_, all_one_tmp_;
};

} //namespace micronet

#endif // CONVOLUTION_H
