/**
 * @file pooling.cpp
 * @auther yefajie
 * @data 2018/6/23
 **/
#include <limits>
#include <cmath>
#include <iostream>
#include <thread>
#include <omp.h>
#include "pooling.h"
#include "util.h"

namespace micronet {

Pooling::Pooling(int kernel_h, int kernel_w, int stride_h, int stride_w, const string& padding,
                 const string& pooling, const string& layer_name):
                 Layer(layer_name, "Pooling"), mask_(new Chunk()) {
    if (padding != "same" && padding != "valid") {
        cout << "Padding must be same or valid !" << endl;
        exit(1);
    }
    if (pooling != "max" && pooling != "avg" && pooling != "random") {
        cout << "pooling must be max, avg or random !" << endl;
        exit(1);
    }
    str_hps_["padding"] = padding;
    int_hps_["kernel_h"] = kernel_h;
    int_hps_["kernel_w"] = kernel_w;
    int_hps_["stride_h"] = stride_h;
    int_hps_["stride_w"] = stride_w;
    str_hps_["pooling"] = pooling;
    cout << "Initialize pooling layer: " << layer_name << " done..." << endl;
}

chunk_ptr Pooling::operator()(chunk_ptr& in_chunk) {
    chunks_in_ = {in_chunk};
    pad_inference();
    chunk_ptr out_chunk = make_shared<Chunk>(shape_inference());
    chunks_out_ = {out_chunk};

    layer_ptr layer = make_shared<Pooling>(*this);
    in_chunk->in_layers_.push_back(layer);
    out_chunk->out_layer_ = layer;

    return out_chunk;
}

void Pooling::forward(bool is_train) {
    Timer timer;
    int kernel_h = int_hps_["kernel_h"];
    int kernel_w = int_hps_["kernel_w"];
    int pad_h = int_hps_["pad_h"];
    int pad_w = int_hps_["pad_w"];
    int stride_h = int_hps_["stride_h"];
    int stride_w = int_hps_["stride_w"];
    string pooling = str_hps_["pooling"];

    chunk_ptr in_chunk = chunks_in_[0];
    chunk_ptr out_chunk = chunks_out_[0];

    vector<int> out_shape = shape_inference();
    int input_h = in_chunk->height();
    int input_w = in_chunk->width();
    int output_h = out_shape[2];
    int output_w = out_shape[3];
    int num = in_chunk->num();
    int channels = in_chunk->channels();
    out_chunk->reshape(out_shape);
    if (pooling == "max" || pooling == "random") {
        mask_->reshape(out_shape);
    }

    const float* input_data = in_chunk->const_data();
    float* output_data = out_chunk->data();

    if (pooling == "max") {
        float* mask_data = mask_->data();
        #pragma omp parallel for
        for (int n = 0; n < num; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int oh = 0; oh < output_h; ++oh) {
                    for (int ow = 0; ow <output_w; ++ow) {
                        int hstart = oh * stride_h - pad_h;
                        int wstart = ow * stride_w - pad_w;
                        int hend = min(hstart + kernel_h, input_h);
                        int wend = min(wstart + kernel_w, input_w);
                        hstart = max(hstart, 0);
                        wstart = max(wstart, 0);
                        const int oindex = out_chunk->offset(n, c, oh, ow);
                        output_data[oindex] = -numeric_limits<float>::max();
                        mask_data[oindex] = -1;
                        for (int ih = hstart; ih < hend; ++ih) {
                            for (int iw = wstart; iw < wend; iw++) {
                                const int iindex = in_chunk->offset(n, c, ih, iw);
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
        #pragma omp parallel for
        for (int n = 0; n < num; ++n) {
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
                        const int oindex = out_chunk->offset(n, c, oh, ow);
                        output_data[oindex] = 0;
                        for (int ih = hstart; ih < hend; ++ih) {
                            for (int iw = wstart; iw < wend; iw++) {
                                const int iindex = in_chunk->offset(n, c, ih, iw);
                                output_data[oindex] += input_data[iindex];
                            }
                        }
                        output_data[oindex] /= max(1, pooling_size);
                    }
                }
            }
        }
    } else if (pooling == "random") {
        UniformGenerator generator(0.0f, 1.0f);
        float* mask_data = mask_->data();
        #pragma omp parallel for
        for (int n = 0; n < num; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int oh = 0; oh < output_h; ++oh) {
                    for (int ow = 0; ow <output_w; ++ow) {
                        int hstart = oh * stride_h - pad_h;
                        int wstart = ow * stride_w - pad_w;
                        int hend = min(hstart + kernel_h, input_h);
                        int wend = min(wstart + kernel_w, input_w);
                        hstart = max(hstart, 0);
                        wstart = max(wstart, 0);
                        const int oindex = out_chunk->offset(n, c, oh, ow);
                        const int pooling_size = (hend - hstart) * (wend - wstart);

                        if (is_train) {
                            const int rand_tmp = static_cast<int>(pooling_size*generator());
                            const int ih = hstart + rand_tmp / (wend - wstart);
                            const int iw = wstart + rand_tmp % (wend - wstart);
                            const int iindex = in_chunk->offset(n, c, ih, iw);
                            output_data[oindex] = input_data[iindex];
                            mask_data[oindex] = iindex;
                        } else {
                            output_data[oindex] = 0;
                            for (int ih = hstart; ih < hend; ++ih) {
                                for (int iw = wstart; iw < wend; iw++) {
                                    const int iindex = in_chunk->offset(n, c, ih, iw);
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
    //cout << "pooling forward time:" << timer.elapsed()*1000 << endl;
    //exit(0);
    gradient_reset();
    //cout << "pooling forward" << endl;
}

void Pooling::backward() {
    Timer timer;
    int kernel_h = int_hps_["kernel_h"];
    int kernel_w = int_hps_["kernel_w"];
    int pad_h = int_hps_["pad_h"];
    int pad_w = int_hps_["pad_w"];
    int stride_h = int_hps_["stride_h"];
    int stride_w = int_hps_["stride_w"];
    string pooling = str_hps_["pooling"];

    chunk_ptr in_chunk = chunks_in_[0];
    chunk_ptr out_chunk = chunks_out_[0];

    vector<int> out_shape = shape_inference();
    int input_h = in_chunk->height();
    int input_w = in_chunk->width();
    int output_h = out_shape[2];
    int output_w = out_shape[3];
    int num = in_chunk->num();
    int channels = in_chunk->channels();

    const float* output_diff = out_chunk->const_diff();
    float* input_diff = in_chunk->diff();
    if (pooling == "max" || pooling == "random") {
        const float* mask_data = mask_->const_data();
        #pragma omp parallel for
        for (int n = 0; n < num; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int oh = 0; oh < output_h; ++oh) {
                    for (int ow = 0; ow <output_w; ++ow) {
                        const int oindex = out_chunk->offset(n, c, oh, ow);
                        const int iindex = static_cast<int>(mask_data[oindex]);
                        input_diff[iindex] += output_diff[oindex];
                    }
                }
            }
        }
    } else if (pooling == "avg") {
        #pragma omp parallel for
        for (int n = 0; n < num; ++n) {
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
                        const int oindex = out_chunk->offset(n, c, oh, ow);
                        for (int ih = hstart; ih < hend; ++ih) {
                            for (int iw = wstart; iw < wend; iw++) {
                                const int iindex = in_chunk->offset(n, c, ih, iw);
                                input_diff[iindex] += output_diff[oindex] / pooling_size;
                            }
                        }
                    }
                }
            }
        }
    }
    //cout << "pooling back time:" << timer.elapsed()*1000 << endl;
    //exit(0);
}

void Pooling::pad_inference() {
    if (str_hps_["padding"] == "valid") {
        int_hps_["pad_h"] = 0;
        int_hps_["pad_w"] = 0;
    } else if (str_hps_["padding"] == "same") {
        int kernel_h = int_hps_["kernel_h"];
        int kernel_w = int_hps_["kernel_w"];
        int stride_h = int_hps_["stride_h"];
        int stride_w = int_hps_["stride_w"];

        int input_h = chunks_in_[0]->height();
        int input_w = chunks_in_[0]->width();
        int output_h = std::ceil(float(input_h) / stride_h);
        int output_w = std::ceil(float(input_w) / stride_w);
        int_hps_["pad_h"] = (output_h * stride_h + kernel_h - input_h) / 2;
        int_hps_["pad_w"] = (output_w * stride_w + kernel_w - input_w) / 2;
    }
}

vector<int> Pooling::shape_inference() {
    int kernel_h = int_hps_["kernel_h"];
    int kernel_w = int_hps_["kernel_w"];
    int pad_h = int_hps_["pad_h"];
    int pad_w = int_hps_["pad_w"];
    int stride_h = int_hps_["stride_h"];
    int stride_w = int_hps_["stride_w"];

    int input_h = chunks_in_[0]->height();
    int input_w = chunks_in_[0]->width();
    int output_h = (input_h + 2*pad_h - kernel_h) / stride_h + 1;
    int output_w = (input_w + 2*pad_w - kernel_w) / stride_w + 1;
    int num = chunks_in_[0]->num();
    int output_channels = chunks_in_[0]->channels();

    return {num, output_channels, output_h, output_w};
}

} // namespace micronet
