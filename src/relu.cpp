/**
 * @file relu.cpp
 * @auther yefajie
 * @data 2018/6/23
 **/
#include "relu.h"

ReLU::ReLU(const string& layer_name) {
    layer_name_ = layer_name;
    cout << "Initialize relu layer: " << layer_name_ << " done..." << endl;
}

void ReLU::set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks) {
    in_chunks_ = in_chunks;
    out_chunks_ = out_chunks;
}

void ReLU::forward(const vector<Chunk*>& input, const vector<Chunk*>& output) {
    output[0]->reshape(input[0]->shape());
    const float* input_data = input[0]->const_data();
    float* output_data = output[0]->data();
    for (int i = 0; i < input[0]->count(); ++i) {
        output_data[i] = std::max(float(0), input_data[i]);
    }
}

void ReLU::backward(const vector<Chunk*>& input, const vector<Chunk*>& output) {
    const float* input_data = input[0]->const_data();
    const float* output_diff = output[0]->const_diff();
    float* input_diff = input[0]->diff();
    for (int i = 0; i < input[0]->count(); ++i) {
        input_diff[i] = (input_data[i] > 0) * output_diff[i];
    }
}
