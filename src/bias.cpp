/**
 * @file bias.cpp
 * @auther yefajie
 * @data 2018/6/23
 **/
#include <iostream>
#include "bias.h"

Bias::Bias(int input_channels, const string& layer_name, float bias_value) {
    params_.push_back(make_shared<Chunk>(1, input_channels, 1, 1));
    initialize(bias_value);
    layer_name_ = layer_name;
    cout << "Initialize bias layer: " << layer_name_ << " done..." << endl;
}

void Bias::set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks) {
    in_chunks_ = in_chunks;
    out_chunks_ = out_chunks;
}

void Bias::forward(const vector<Chunk*>& input, const vector<Chunk*>& output) {
    int num = input[0]->num();
    int input_channels = input[0]->channels();
    int input_h = input[0]->height();
    int input_w = input[0]->width();
    output[0]->reshape(num, input_channels, input_h, input_w);

    if (params_[0]->channels() != input_channels) {
        cout << "Bias Error: bias channels must equal input channels(" << params_[0]->channels() << "!=" << input_channels << ")";
        exit(1);
    }

    const float* input_data = input[0]->const_data();
    const float* bias_data = params_[0]->const_data();
    float* output_data = output[0]->data();
    for (int i = 0; i < output[0]->count(); ++i) {
        output_data[i] = float(0);
    }

    for (int n = 0; n < num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
            const int bindex = params_[0]->offset(0, c, 0, 0);
            for (int h = 0; h < input_h; ++h) {
                for (int w = 0; w < input_w; ++w) {
                    const int oindex = output[0]->offset(n, c, h, w);
                    output_data[oindex] = input_data[oindex] + bias_data[bindex];
                }
            }
        }
    }
}

void Bias::backward(const vector<Chunk*>& input, const vector<Chunk*>& output) {
    int num = input[0]->num();
    int input_channels = input[0]->channels();
    int input_h = input[0]->height();
    int input_w = input[0]->width();

    const float* output_diff = output[0]->const_diff();
    float* input_diff = input[0]->diff();
    float* bias_diff = params_[0]->diff();
    for (int i = 0; i < input[0]->count(); ++i) {
        input_diff[i] = float(0);
    }
    for (int i = 0; i < params_[0]->count(); ++i) {
        bias_diff[i] = float(0);
    }

    for (int n = 0; n < num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
            const int bindex = params_[0]->offset(0, c, 0, 0);
            for (int h = 0; h < input_h; ++h) {
                for (int w = 0; w < input_w; ++w) {
                    const int oindex = output[0]->offset(n, c, h, w);
                    bias_diff[bindex] += output_diff[oindex];
                    input_diff[oindex] += output_diff[oindex];
                }
            }
        }
    }
}

void Bias::initialize(float bias_value) {
    float* data = params_[0]->data();
    for (int i = 0; i < params_[0]->count(); ++i) {
        data[i] = bias_value;
    }
}
