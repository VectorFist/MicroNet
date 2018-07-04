/**
 * @file dense.cpp
 * @auther yefajie
 * @data 2018/6/23
 **/
#include <random>
#include <chrono>
#include <iostream>
#include <thread>
#include "dense.h"

Dense::Dense(int in_dim, int out_dim, const string& layer_name, float mean, float stddev, float bias_value) {
    params_.push_back(make_shared<Chunk>(in_dim, out_dim, 1, 1));
    params_.push_back(make_shared<Chunk>(1, out_dim, 1, 1));
    initialize(mean, stddev, bias_value);
    layer_name_ = layer_name;
    cout << "Initialize dense layer: " << layer_name_ << " done..." << endl;
}

void Dense::set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks) {
    in_chunks_ = in_chunks;
    out_chunks_ = out_chunks;
}

void __forward__(Chunk* input, Chunk* output, shared_ptr<Chunk>* weights, shared_ptr<Chunk>* bias,
                 int start_n, int end_n, int out_dim, int in_dim) {
    const float* input_data = input->const_data();
    const float* weights_data = (*weights)->const_data();
    const float* bias_data = (*bias)->const_data();
    float* output_data = output->data();

    for (int n = start_n; n < end_n; ++n) {
        for (int od = 0; od < out_dim; ++od) {
            const int oindex = output->offset(n, od, 0, 0);
            const int bindex = (*bias)->offset(0, od, 0, 0);
            output_data[oindex] = 0;
            for (int id = 0; id < in_dim; ++id) {
                const int iindex = n * in_dim + id;
                const int windex = (*weights)->offset(id, od, 0, 0);
                output_data[oindex] += input_data[iindex] * weights_data[windex];
            }
            output_data[oindex] += bias_data[bindex];
        }
    }
}

void Dense::forward(const vector<Chunk*>& input, const vector<Chunk*>& output) {
    int num = input[0]->num();
    int input_channels = input[0]->channels();
    int input_h = input[0]->height();
    int input_w = input[0]->width();
    int in_dim = input_channels * input_h * input_w;
    int out_dim = params_[0]->channels();
    output[0]->reshape(num, out_dim, 1, 1);

    if (params_[0]->num() != in_dim) {
        cout << "Dense Error: weight num must match input dim(" << params_[0]->num() << "!=" << in_dim << ")";
        exit(1);
    }
    if (num < 4) {
        __forward__(input[0], output[0], &params_[0], &params_[1],
                    0, num, out_dim, in_dim);
    } else {
        vector<thread> threads;
        int block = int(ceil(num / 4.0));
        for (int i = 0; i < 4; ++i) {
            threads.push_back(thread(__forward__, input[0], output[0], &params_[0], &params_[1], i*block, min(num, (i+1)*block),
                                     out_dim, in_dim));
        }
        for (int i = 0; i < 4; ++i) {
            threads[i].join();
        }
    }
}

void Dense::backward(const vector<Chunk*>& input, const vector<Chunk*>& output) {
    int num = input[0]->num();
    int input_channels = input[0]->channels();
    int input_h = input[0]->height();
    int input_w = input[0]->width();
    int in_dim = input_channels * input_h * input_w;
    int out_dim = params_[0]->channels();

    const float* input_data = input[0]->const_data();
    const float* weights_data = params_[0]->const_data();
    const float* output_diff = output[0]->const_diff();
    float* input_diff = input[0]->diff();
    float* weights_diff = params_[0]->diff();
    float* bias_diff = params_[1]->diff();
    for (int i = 0; i < input[0]->count(); ++i) {
        input_diff[i] = float(0);
    }
    for (int i = 0; i < params_[0]->count(); ++i) {
        weights_diff[i] = float(0);
    }
    for (int i = 0; i < params_[1]->count(); ++i) {
        bias_diff[i] = float(0);
    }

    for (int n = 0; n < num; ++n) {
        for (int od = 0; od < out_dim; ++od) {
            const int oindex = output[0]->offset(n, od, 0, 0);
            const int bindex = params_[1]->offset(0, od, 0, 0);
            for (int id = 0; id <in_dim; ++id) {
                const int iindex = n * in_dim + id;
                const int windex = params_[0]->offset(id, od, 0, 0);
                input_diff[iindex] += output_diff[oindex] * weights_data[windex];
                weights_diff[windex] += output_diff[oindex] * input_data[iindex];
            }
            bias_diff[bindex] += output_diff[oindex];
        }
    }
}

void Dense::initialize(float mean, float stddev, float bias_value) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(mean, stddev);
    float* weights_data = params_[0]->data();
    float* bias_data = params_[1]->data();
    for (int i = 0; i < params_[0]->count(); ++i) {
        float rnd = distribution(generator);
        int trial = 0;
        while ((rnd < (mean - 2 * stddev) || rnd > (mean + 2 * stddev)) && trial < 5) {
            rnd = distribution(generator);
            trial++;
        }
        weights_data[i] = rnd;
    }
    for (int i = 0; i < params_[1]->count(); ++i) {
        bias_data[i] = bias_value;
    }
}
