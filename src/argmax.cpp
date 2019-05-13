/**
 * @file argmax.cpp
 * @auther yefajie
 * @data 2018/6/24
 **/
#include <utility>
#include <algorithm>
#include <omp.h>
#include "argmax.h"

namespace micronet {

ArgMax::ArgMax(const string& layer_name): Layer(layer_name, "ArgMax") {
}

chunk_ptr ArgMax::operator()(const chunk_ptr& in_chunk) {
    chunks_in_ = {in_chunk};
    vector<int> out_shape = shape_inference();
    chunk_ptr out_chunk = make_shared<Chunk>(out_shape);
    chunks_out_ = {out_chunk};

    layer_ptr layer = make_shared<ArgMax>(*this);
    in_chunk->in_layers_.push_back(layer);
    out_chunk->out_layer_ = layer;

    return out_chunk;
}

void ArgMax::forward(bool is_train) {
    chunk_ptr in_chunk = chunks_in_[0];
    chunk_ptr out_chunk = chunks_out_[0];

    int num = in_chunk->num();
    int input_channels = in_chunk->channels();
    int input_h = in_chunk->height();
    int input_w = in_chunk->width();
    out_chunk->reshape(num, 1, input_h, input_w);

    const float* input_data = in_chunk->const_data();
    float* output_data = out_chunk->data();
    //#pragma omp parallel for
    for (int n = 0; n < num; ++n) {
        for (int h = 0; h < input_h; ++h) {
            for (int w = 0; w < input_w; ++w) {
                vector<pair<float, float> > arg_index;
                for (int c = 0; c < input_channels; ++c) {
                    const int iindex = in_chunk->offset(n, c, h, w);
                    arg_index.push_back(std::make_pair(float(c), input_data[iindex]));
                }
                sort(arg_index.begin(), arg_index.end(),
                     [](const pair<float, float>& p1, const pair<float, float>& p2){return p1.second > p2.second;});
                const int oindex = out_chunk->offset(n, 0, h, w);
                output_data[oindex] = arg_index[0].first;
            }
        }
    }
}

vector<int> ArgMax::shape_inference() {
    int num = chunks_in_[0]->num();
    int height = chunks_in_[0]->height();
    int width = chunks_in_[0]->width();
    return {num, 1, height, width};
}

} // namespace micronet
