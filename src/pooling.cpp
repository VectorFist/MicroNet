/**
 * @file pooling.cpp
 * @auther yefajie
 * @data 2018/6/23
 **/
#include <limits>
#include <cmath>
#include <iostream>
#include <thread>
#include "pooling.h"

Pooling::Pooling(int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h,
                 int stride_w, const string& layer_name, const string& pooling) {
    kernel_h_ = kernel_h;
    kernel_w_ = kernel_w;
    pad_h_ = pad_h;
    pad_w_ = pad_w;
    stride_h_ = stride_h;
    stride_w_ = stride_w;
    pooling_ = pooling;
    layer_name_ = layer_name;
    cout << "Initialize pooling layer: " << layer_name_ << " done..." << endl;
}

void Pooling::set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks) {
    in_chunks_ = in_chunks;
    out_chunks_ = out_chunks;
}

void __forward__(Chunk* input, Chunk* output, Chunk* mask, string pooling, int start_n, int end_n,
                 int channels, int output_h, int output_w,
                 int input_h, int input_w, int kernel_h, int kernel_w, int pad_h, int pad_w,
                 int stride_h, int stride_w) {
    const float* input_data = input->const_data();
    float* output_data = output->data();

    if (pooling == "max") {
        float* mask_data = mask->data();
        for (int n = start_n; n < end_n; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int oh = 0; oh < output_h; ++oh) {
                    for (int ow = 0; ow <output_w; ++ow) {
                        int hstart = oh * stride_h - pad_h;
                        int wstart = ow * stride_w - pad_w;
                        int hend = min(hstart + kernel_h, input_h);
                        int wend = min(wstart + kernel_w, input_w);
                        hstart = max(hstart, 0);
                        wstart = max(wstart, 0);
                        const int oindex = output->offset(n, c, oh, ow);
                        output_data[oindex] = -numeric_limits<float>::min();
                        mask_data[oindex] = -1;
                        for (int ih = hstart; ih < hend; ++ih) {
                            for (int iw = wstart; iw < wend; iw++) {
                                const int iindex = input->offset(n, c, ih, iw);
                                if (input_data[iindex] > output_data[oindex]) {
                                    output_data[oindex] = input_data[iindex];
                                    mask_data[oindex] = iindex;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if (pooling == "avg") {
        for (int n = start_n; n < end_n; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int oh = 0; oh < output_h; ++oh) {
                    for (int ow = 0; ow <output_w; ++ow) {
                        int hstart = oh * stride_h - pad_h;
                        int wstart = ow * stride_w - pad_w;
                        int hend = min(hstart + kernel_h, input_h);
                        int wend = min(wstart + kernel_w, input_w);
                        hstart = max(hstart, 0);
                        wstart = max(wstart, 0);
                        const int pooling_size = (hend - hstart) * (wend - wstart);
                        const int oindex = output->offset(n, c, oh, ow);
                        output_data[oindex] = 0;
                        for (int ih = hstart; ih < hend; ++ih) {
                            for (int iw = wstart; iw < wend; iw++) {
                                const int iindex = input->offset(n, c, ih, iw);
                                output_data[oindex] += input_data[iindex];
                            }
                        }
                        output_data[oindex] /= max(1, pooling_size);
                    }
                }
            }
        }
    }
}

void Pooling::forward(const vector<Chunk*>& input, const vector<Chunk*>& output) {
    int input_h = input[0]->height();
    int input_w = input[0]->width();
    int output_h = static_cast<int>(ceil(static_cast<float>(input_h + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
    int output_w = static_cast<int>(ceil(static_cast<float>(input_w + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
    int num = input[0]->num();
    int channels = input[0]->channels();
    output[0]->reshape(num, channels, output_h, output_w);
    if (pooling_ == "max") {
        mask_.reshape(num, channels, output_h, output_w);
    }

    if (num < 4) {
        __forward__(input[0], output[0], &mask_, pooling_, 0, num,
                    channels, output_h, output_w,
                    input_h, input_w, kernel_h_, kernel_w_, pad_h_, pad_w_,
                    stride_h_, stride_w_);
    } else {
        vector<thread> threads;
        int block = int(ceil(num / 4.0));
        for (int i = 0; i < 4; ++i) {
            threads.push_back(thread(__forward__, input[0], output[0], &mask_, pooling_, i*block, min(num, (i+1)*block),
                                     channels, output_h, output_w,
                                     input_h, input_w, kernel_h_, kernel_w_, pad_h_, pad_w_,
                                     stride_h_, stride_w_));
        }
        for (int i = 0; i < 4; ++i) {
            threads[i].join();
        }
    }
}

void Pooling::backward(const vector<Chunk*>& input, const vector<Chunk*>& output) {
    int input_h = input[0]->height();
    int input_w = input[0]->width();
    int output_h = static_cast<int>(ceil(static_cast<float>(input_h + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
    int output_w = static_cast<int>(ceil(static_cast<float>(input_w + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
    int num = input[0]->num();
    int channels = input[0]->channels();

    const float* output_diff = output[0]->const_diff();
    float* input_diff = input[0]->diff();
    for (int i = 0; i < input[0]->count(); ++i) {
        input_diff[i] = 0;
    }

    if (pooling_ == "max") {
        const float* mask_data = mask_.const_data();
        for (int n = 0; n < num; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int oh = 0; oh < output_h; ++oh) {
                    for (int ow = 0; ow <output_w; ++ow) {
                        const int oindex = output[0]->offset(n, c, oh, ow);
                        const int iindex = static_cast<int>(mask_data[oindex]);
                        input_diff[iindex] += output_diff[oindex];
                    }
                }
            }
        }
    } else if (pooling_ == "avg") {
        for (int n = 0; n < num; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int oh = 0; oh < output_h; ++oh) {
                    for (int ow = 0; ow <output_w; ++ow) {
                        int hstart = oh * stride_h_ - pad_h_;
                        int wstart = ow * stride_w_ - pad_w_;
                        int hend = min(hstart + kernel_h_, input_h);
                        int wend = min(wstart + kernel_w_, input_w);
                        hstart = max(hstart, 0);
                        wstart = max(wstart, 0);
                        const int pooling_size = (hend - hstart) * (wend - wstart);
                        const int oindex = output[0]->offset(n, c, oh, ow);
                        for (int ih = hstart; ih < hend; ++ih) {
                            for (int iw = wstart; iw < wend; iw++) {
                                const int iindex = input[0]->offset(n, c, ih, iw);
                                input_diff[iindex] += output_diff[oindex] / pooling_size;
                            }
                        }
                    }
                }
            }
        }
    }
}
