/**
 * @file convoluntion.cpp
 * @auther yefajie
 * @data 2018/6/23
 **/
#include <random>
#include <chrono>
#include <iostream>
#include <thread>
#include "convolution.h"

Convolution::Convolution(int kernel_h, int kernel_w, int pad_h, int pad_w,
                         int stride_h, int stride_w, int input_channels, int output_channels,
                         const string& layer_name, float mean, float stddev, float bias_value) {
    kernel_h_ = kernel_h;
    kernel_w_ = kernel_w;
    pad_h_ = pad_h;
    pad_w_ = pad_w;
    stride_h_ = stride_h;
    stride_w_ = stride_w;
    input_channels_ = input_channels;
    output_channels_ = output_channels;
    params_.push_back(make_shared<Chunk>(input_channels, output_channels, kernel_h, kernel_w));
    params_.push_back(make_shared<Chunk>(1, output_channels, 1, 1));
    initialize(mean, stddev, bias_value);
    layer_name_ = layer_name;
    cout << "Initialize conv layer: " << layer_name_ << " done..." << endl;
}

void Convolution::set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks) {
    in_chunks_ = in_chunks;
    out_chunks_ = out_chunks;
}

void __forward__(Chunk* input, Chunk* output, shared_ptr<Chunk>* weights, shared_ptr<Chunk>* bias, int start_n, int end_n,
                 int output_channels, int output_h, int output_w, int input_channels,
                 int input_h, int input_w, int kernel_h, int kernel_w, int pad_h, int pad_w,
                 int stride_h, int stride_w) {
    const float* input_data = input->const_data();
    const float* weights_data = (*weights)->const_data();
    const float* bias_data = (*bias)->const_data();
    float* output_data = output->data();
    for (int n = start_n; n < end_n; ++n) {
        for (int oc = 0; oc < output_channels; ++oc) {
            const int bindex = (*bias)->offset(0, oc, 0, 0);
            for (int oh = 0; oh < output_h; ++oh) {
                for (int ow = 0; ow < output_w; ++ow) {
                    int hstart = oh * stride_h - pad_h;
                    int wstart = ow * stride_w - pad_w;
                    int hend = min(hstart + kernel_h, input_h);
                    int wend = min(wstart + kernel_w, input_w);
                    hstart = max(hstart, 0);
                    wstart = max(wstart, 0);
                    const int oindex = output->offset(n, oc, oh, ow);
                    output_data[oindex] = 0;
                    for (int ic = 0; ic < input_channels; ++ic) {
                        for (int ih = hstart; ih < hend; ++ih) {
                            for (int iw = wstart; iw < wend; ++iw) {
                                const int iindex = input->offset(n, ic, ih, iw);
                                const int windex = (*weights)->offset(ic, oc, ih - hstart, iw - wstart);
                                output_data[oindex] += input_data[iindex] * weights_data[windex];
                            }
                        }
                    }
                    output_data[oindex] += bias_data[bindex];
                }
            }
        }
    }
}

void Convolution::forward(const vector<Chunk*>& input, const vector<Chunk*>& output) {
    int input_h = input[0]->height();
    int input_w = input[0]->width();
    int output_h = static_cast<int>(ceil(static_cast<float>(input_h + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
    int output_w = static_cast<int>(ceil(static_cast<float>(input_w + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
    int num = input[0]->num();
    int input_channels = input[0]->channels();
    int output_channels = params_[0]->shape(1);
    output[0]->reshape(num, output_channels, output_h, output_w);

    if (input_channels != input_channels_) {
        cout << "Conv Error: weight num must match input channels(" << input_channels_ << "!=" << input_channels << ")";
        exit(1);
    }

    if (num < 4) {
        __forward__(input[0], output[0], &params_[0], &params_[1], 0, num,
                    output_channels, output_h, output_w, input_channels,
                    input_h, input_w, kernel_h_, kernel_w_, pad_h_, pad_w_,
                    stride_h_, stride_w_);
    } else {
        vector<thread> threads;
        int block = int(ceil(num / 4.0));
        for (int i = 0; i < 4; ++i) {
            threads.push_back(thread(__forward__, input[0], output[0], &params_[0], &params_[1], i*block, min(num, (i+1)*block),
                                     output_channels, output_h, output_w, input_channels,
                                     input_h, input_w, kernel_h_, kernel_w_, pad_h_, pad_w_,
                                     stride_h_, stride_w_));
        }
        for (int i = 0; i < 4; ++i) {
            threads[i].join();
        }
    }
}

void Convolution::backward(const vector<Chunk*>& input, const vector<Chunk*>& output) {
    int input_h = input[0]->height();
    int input_w = input[0]->width();
    int output_h = output[0]->height();
    int output_w = output[0]->width();
    int num = input[0]->num();
    int input_channels = input[0]->channels();
    int output_channels = output[0]->channels();

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
    for (int i = 0; i <params_[1]->count(); ++i) {
        bias_diff[i] = float(0);
    }

    for (int n = 0; n < num; ++n) {
        for (int oc = 0; oc < output_channels; ++oc) {
            const int bindex = params_[1]->offset(0, oc, 0, 0);
            for (int oh = 0; oh < output_h; ++oh) {
                for (int ow = 0; ow < output_w; ++ow) {
                    int hstart = oh * stride_h_ - pad_h_;
                    int wstart = ow * stride_w_ - pad_w_;
                    int hend = min(hstart + kernel_h_, input_h);
                    int wend = min(wstart + kernel_w_, input_w);
                    hstart = max(hstart, 0);
                    wstart = max(wstart, 0);
                    const int oindex = output[0]->offset(n, oc, oh, ow);
                    for (int ic = 0; ic < input_channels; ++ic) {
                        for (int ih = hstart; ih < hend; ++ih) {
                            for (int iw = wstart; iw < wend; ++iw) {
                                const int iindex = input[0]->offset(n, ic, ih, iw);
                                const int windex = params_[0]->offset(ic, oc, ih - hstart, iw - wstart);
                                input_diff[iindex] += output_diff[oindex] * weights_data[windex];
                                weights_diff[windex] += output_diff[oindex] * input_data[iindex];
                            }
                        }
                    }
                    bias_diff[bindex] += output_diff[oindex];
                }
            }
        }
    }
}

void Convolution::initialize(float mean, float stddev, float bias_value) {
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
