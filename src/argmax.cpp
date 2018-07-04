/**
 * @file argmax.cpp
 * @auther yefajie
 * @data 2018/6/24
 **/
#include <utility>
#include <algorithm>
#include "argmax.h"


ArgMax::ArgMax(const string& layer_name) {
    layer_name_ = layer_name;
}

void ArgMax::set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks) {
    in_chunks_ = in_chunks;
    out_chunks_ = out_chunks;
}

void ArgMax::forward(const vector<Chunk*>& input, const vector<Chunk*>& output) {
    int num = input[0]->num();
    int input_channels = input[0]->channels();
    int input_h = input[0]->height();
    int input_w = input[0]->width();
    output[0]->reshape(num, 1, input_h, input_w);

    const float* input_data = input[0]->const_data();
    float* output_data = output[0]->data();
    for (int n = 0; n < num; ++n) {
        for (int h = 0; h < input_h; ++h) {
            for (int w = 0; w < input_w; ++w) {
                vector<pair<float, float> > arg_index;
                for (int c = 0; c < input_channels; ++c) {
                    const int iindex = input[0]->offset(n, c, h, w);
                    arg_index.push_back(std::make_pair(float(c), input_data[iindex]));
                }
                sort(arg_index.begin(), arg_index.end(),
                     [](const pair<float, float>& p1, const pair<float, float>& p2){return p1.second > p2.second;});
                const int oindex = output[0]->offset(n, 0, h, w);
                output_data[oindex] = arg_index[0].first;
            }
        }
    }
}
