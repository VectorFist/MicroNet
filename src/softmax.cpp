/**
 * @file softmax.cpp
 * @auther yefajie
 * @data 2018/6/24
 **/
#include <cmath>
#include "softmax.h"

Softmax::Softmax(const string& layer_name) {
    layer_name_ = layer_name;
}

void Softmax::set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks) {
    in_chunks_ = in_chunks;
    out_chunks_ = out_chunks;
}

void Softmax::forward(const vector<Chunk*>& input, const vector<Chunk*>& output) {
    int num = input[0]->num();
    int input_channels = input[0]->channels();
    int input_h = input[0]->height();
    int input_w = input[0]->width();
    output[0]->reshape(num, input_channels, input_h, input_w);

    const float* input_data = input[0]->const_data();
    float* output_data = output[0]->data();
    for (int i = 0; i < output[0]->count(); ++i) {
        output_data[i] = float(0);
    }

    for (int n = 0; n < num; ++n) {
        for (int h = 0; h < input_h; ++h) {
            for (int w = 0; w < input_w; ++w) {
                float exp_sum = 0;
                for (int c = 0; c < input_channels; ++c) {
                    const int iindex = input[0]->offset(n, c, h, w);
                    exp_sum += min(exp(input_data[iindex]), FLT_MAX);
                }
                for (int c = 0;  c < input_channels; ++c) {
                    const int index = input[0]->offset(n, c, h, w);
                    output_data[index] = exp(input_data[index]) / exp_sum;
                }
            }
        }
    }
}
