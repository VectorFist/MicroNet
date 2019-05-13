/**
 * @file softmax.cpp
 * @auther yefajie
 * @data 2018/6/24
 **/
#include <cmath>
#include <omp.h>
#include <string.h>
#include "softmax.h"

namespace micronet {

Softmax::Softmax(const string& layer_name): Layer(layer_name, "Softmax") {
}

chunk_ptr Softmax::operator()(chunk_ptr& in_chunk) {
    chunks_in_ = {in_chunk};
    chunk_ptr out_chunk = make_shared<Chunk>(shape_inference());
    chunks_out_ = {out_chunk};

    layer_ptr layer = make_shared<Softmax>(*this);
    in_chunk->in_layers_.push_back(layer);
    out_chunk->out_layer_ = layer;

    return out_chunk;
}

void Softmax::forward(bool is_train) {
    chunk_ptr in_chunk = chunks_in_[0];
    chunk_ptr out_chunk = chunks_out_[0];
    int num = in_chunk->num();
    int input_channels = in_chunk->channels();
    int input_h = in_chunk->height();
    int input_w = in_chunk->width();
    out_chunk->reshape(num, input_channels, input_h, input_w);

    const float* input_data = in_chunk->const_data();
    float* output_data = out_chunk->data();
    memset(output_data, 0, out_chunk->count()*sizeof(float));

    //#pragma omp parallel for
    for (int n = 0; n < num; ++n) {
        for (int h = 0; h < input_h; ++h) {
            for (int w = 0; w < input_w; ++w) {
                float exp_sum = 0;
                for (int c = 0; c < input_channels; ++c) {
                    const int iindex = in_chunk->offset(n, c, h, w);
                    exp_sum += min(exp(input_data[iindex]), FLT_MAX);
                }
                for (int c = 0;  c < input_channels; ++c) {
                    const int index = in_chunk->offset(n, c, h, w);
                    output_data[index] = exp(input_data[index]) / exp_sum;
                }
            }
        }
    }
    //cout << "softmax forward" << endl;
}

vector<int> Softmax::shape_inference() {
    return chunks_in_[0]->shape();
}

} // namespace micronet
