/**
 * @file dense.cpp
 * @auther yefajie
 * @data 2018/6/23
 **/
#include <random>
#include <chrono>
#include <iostream>
#include <thread>
#include <string.h>

#include "dense.h"
#include "util.h"
#include "math_func.h"

namespace micronet {

Dense::Dense(int out_dim, float mean, float stddev, float bias_value, const string& layer_name):
    Layer(layer_name, "Dense") {
    int_hps_["out_dim"] = out_dim;
    flt_hps_["init_mean"] = mean;
    flt_hps_["init_stddev"] = stddev;
    flt_hps_["init_bias_value"] = bias_value;
    cout << "Initialize dense layer: " << layer_name << " done..." << endl;
}

chunk_ptr Dense::operator()(const chunk_ptr& in_chunk) {
    chunks_in_ = {in_chunk};
    chunk_ptr out_chunk = make_shared<Chunk>(shape_inference());
    chunks_out_ = {out_chunk};

    int in_dim = in_chunk->count() / in_chunk->num();
    params_.push_back(make_shared<Chunk>(in_dim, int_hps_["out_dim"], 1, 1));
    params_.push_back(make_shared<Chunk>(1, int_hps_["out_dim"], 1, 1));
    initialize();

    layer_ptr layer = make_shared<Dense>(*this);
    in_chunk->in_layers_.push_back(layer);
    out_chunk->out_layer_ = layer;

    return out_chunk;
}

void Dense::forward(bool is_train) {
    Timer timer;
    int num = chunks_in_[0]->num();
    int input_channels = chunks_in_[0]->channels();
    int input_h = chunks_in_[0]->height();
    int input_w = chunks_in_[0]->width();
    int in_dim = input_channels * input_h * input_w;
    int out_dim = int_hps_["out_dim"];
    chunks_out_[0]->reshape(num, out_dim, 1, 1);
    all_one_tmp_.reshape(num, 1, 1, 1);
    all_one_tmp_.fill_value(1.0, 1.0);

    const float* input_data = chunks_in_[0]->const_data();
    const float* weight_data = params_[0]->const_data();
    const float* bias_data = params_[1]->const_data();
    const float* all_one_data = all_one_tmp_.const_data();
    float* output_data = chunks_out_[0]->data();
    gemm(0, 0, num, out_dim, in_dim, 1, input_data, in_dim, weight_data, out_dim, 0, output_data, out_dim);
    gemm(0, 0, num, out_dim, 1, 1, all_one_data, 1, bias_data, out_dim, 1, output_data, out_dim);
    /*for (int n = 0; n < num; ++n) {
        add(out_dim, output_data, 1, bias_data, 1, output_data);
        output_data += out_dim;
    }*/
    //cout << "dense forward time:" << timer.elapsed()*1000 << endl;
    //exit(0);
    gradient_reset();
    //cout << "dense forward" << endl;
}

void Dense::backward() {
    Timer timer;
    int num = chunks_in_[0]->num();
    int input_channels = chunks_in_[0]->channels();
    int input_h = chunks_in_[0]->height();
    int input_w = chunks_in_[0]->width();
    int in_dim = input_channels * input_h * input_w;
    int out_dim = int_hps_["out_dim"];

    float* weights_diff = params_[0]->diff();
    float* bias_diff = params_[1]->diff();

    const float* input_data = chunks_in_[0]->const_data();
    const float* weights_data = params_[0]->const_data();
    const float* output_diff = chunks_out_[0]->const_diff();
    const float* all_one_data = all_one_tmp_.const_data();
    float* input_diff = chunks_in_[0]->diff();
    gemm(0, 1, num, in_dim, out_dim, 1, output_diff, out_dim, weights_data, out_dim, 1, input_diff, in_dim);
    gemm(1, 0, in_dim, out_dim, num, 1, input_data, in_dim, output_diff, out_dim, 1, weights_diff, out_dim);
    gemm(1, 0, 1, out_dim, num, 1, all_one_data, 1, output_diff, out_dim, 1, bias_diff, out_dim);
    //cout << bias_diff[1] << endl;
    //exit(0);
    /*for (int n = 0; n < num; ++n) {
        add(out_dim, bias_diff, 1, output_diff, 1, bias_diff);
        output_diff += out_dim;
    }*/
    //cout << "dense back time:" << timer.elapsed()*1000 << endl;
    //exit(0);
}

void Dense::initialize() {
    float* weights_data = params_[0]->data();
    float* bias_data = params_[1]->data();

    normal_random_init(params_[0]->count(), weights_data, flt_hps_["init_mean"], flt_hps_["init_stddev"]);
    constant_init(params_[1]->count(), bias_data, flt_hps_["init_bias_value"]);
}

vector<int> Dense::shape_inference() {
    int num = chunks_in_[0]->num();
    int out_dim = int_hps_["out_dim"];

    return {num, out_dim, 1, 1};
}

} // namespace micronet
