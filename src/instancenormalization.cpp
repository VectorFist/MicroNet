#include <cstring>
#include "omp.h"
#include "instancenormalization.h"
#include "math_func.h"
#include "util.h"

namespace micronet {

InstanceNormalization::InstanceNormalization(const string& layer_name):
    Layer(layer_name, "InstanceNormalization"), mean_(new Chunk()), var_(new Chunk()),
    out_no_shift_(new Chunk()) {
}

chunk_ptr InstanceNormalization::operator()(const chunk_ptr& in_chunk) {
    if (in_chunk->height() * in_chunk->width() == 1) {
        cout << "InstanceNormalization error: inchunk height or width must larger than 1..." << endl;
        exit(1);
    }
    chunks_in_ = {in_chunk};
    chunk_ptr out_chunk = make_shared<Chunk>(shape_inference());
    chunks_out_ = {out_chunk};

    auto gamma = make_shared<Chunk>(1, in_chunk->channels(), 1, 1);
    auto beta = make_shared<Chunk>(1, in_chunk->channels(), 1, 1);
    params_ = {gamma, beta};

    initialize();

    layer_ptr layer = make_shared<InstanceNormalization>(*this);
    in_chunk->in_layers_.push_back(layer);
    out_chunk->out_layer_ = layer;

    return out_chunk;
}

void InstanceNormalization::forward(bool is_train) {
    auto in_chunk = chunks_in_[0];
    auto out_chunk = chunks_out_[0];
    out_chunk->reshape(shape_inference());
    mean_->reshape(in_chunk->num(), in_chunk->channels(), 1, 1);
    var_->reshape(in_chunk->num(), in_chunk->channels(), 1, 1);
    out_no_shift_->reshape(out_chunk->shape());

    auto gamma = params_[0];
    auto beta = params_[1];

    const float* in_data = in_chunk->const_data();
    const float* gamma_data = gamma->const_data();
    const float* beta_data = beta->const_data();
    float* out_no_shift_data = out_no_shift_->data();
    float* out_data = out_chunk->data();
    float* mean_data = mean_->data();
    float* var_data = var_->data();

    const int m = in_chunk->height() * in_chunk->width();
    #pragma omp parallel for
    for (int n = 0; n < in_chunk->num(); ++n) {
        for (int c = 0; c < in_chunk->channels(); ++c) {
            const int mindex = mean_->offset(n, c, 0, 0);
            mean_data[mindex] = 0;
            for (int h = 0; h < in_chunk->height(); ++h) {
                for (int w = 0; w < in_chunk->width(); ++w) {
                    const int iindex = in_chunk->offset(n, c, h, w);
                    mean_data[mindex] += in_data[iindex];
                }
            }
            mean_data[mindex] /= m;
        }
    }
    #pragma omp parallel for
    for (int n = 0; n < in_chunk->num(); ++n) {
        for (int c = 0; c < in_chunk->channels(); ++c) {
            const int vindex = var_->offset(n, c, 0, 0);
            var_data[vindex] = 0;
            for (int h = 0; h < in_chunk->height(); ++h) {
                for (int w = 0; w < in_chunk->width(); ++w) {
                    const int iindex = in_chunk->offset(n, c, h, w);
                    var_data[vindex] += std::pow(in_data[iindex] - mean_data[vindex], 2);
                }
            }
            var_data[vindex] /= m;
        }
    }

    #pragma omp parallel for
    for (int n = 0; n < in_chunk->num(); ++n) {
        for (int c = 0; c < in_chunk->channels(); ++c) {
            const int mindex = mean_->offset(n, c, 0, 0);
            for (int h = 0; h < in_chunk->height(); ++h) {
                for (int w = 0; w < in_chunk->width(); ++w) {
                    const int iindex = in_chunk->offset(n, c, h, w);
                    out_no_shift_data[iindex] = (in_data[iindex] - mean_data[mindex])
                                        / std::sqrt(var_data[mindex] + 1e-8f);
                    out_data[iindex] = gamma_data[c] * out_no_shift_data[iindex] + beta_data[c];
                }
            }
        }
    }

    gradient_reset();
}

void InstanceNormalization::backward() {
    auto in_chunk = chunks_in_[0];
    auto out_chunk = chunks_out_[0];
    auto gamma = params_[0];
    auto beta = params_[1];

    const float* in_data = in_chunk->const_data();
    const float* mean_data = mean_->const_data();
    const float* var_data = var_->const_data();
    const float* gamma_data = gamma->const_data();
    const float* out_diff = out_chunk->const_diff();
    const float* out_no_shift_data = out_no_shift_->const_data();
    float* out_no_shift_diff = out_no_shift_->diff();
    float* in_diff = in_chunk->diff();
    float* mean_diff = mean_->diff();
    float* var_diff = var_->diff();
    float* gamma_diff = gamma->diff();
    float* beta_diff = beta->diff();

    const int m = in_chunk->height() * in_chunk->width();
    #pragma omp parallel for
    for (int n = 0; n < in_chunk->num(); ++n) {
        for (int c = 0; c < in_chunk->channels(); ++c) {
            const int mindex = mean_->offset(n, c, 0, 0);
            mean_diff[mindex] = 0;
            var_diff[mindex] = 0;
            float tmp = 0;
            for (int h = 0; h < in_chunk->height(); ++h) {
                for (int w = 0; w < in_chunk->width(); ++w) {
                    const int oindex = out_chunk->offset(n, c, h, w);
                    out_no_shift_diff[oindex] = gamma_data[c] * out_diff[oindex];
                    var_diff[mindex] += out_no_shift_diff[oindex] * (in_data[oindex]-mean_data[mindex])
                                        * (-0.5) * std::pow(var_data[mindex]+1e-8f, -1.5);
                    mean_diff[mindex] += (-out_no_shift_diff[oindex]) / std::sqrt(var_data[mindex]+1e-8f);
                    tmp += 2 * (mean_data[mindex] - in_data[oindex]);
                }
            }
            mean_diff[mindex] += var_diff[mindex] * tmp / m;
        }
    }

    for (int n = 0; n < in_chunk->num(); ++n) {
        for (int c = 0; c < in_chunk->channels(); ++c) {
            for (int h = 0; h < in_chunk->height(); ++h) {
                for (int w = 0; w < in_chunk->width(); ++w) {
                    const int oindex = out_chunk->offset(n, c, h, w);
                    gamma_diff[c] += out_diff[oindex] * out_no_shift_data[oindex];
                    beta_diff[c] += out_diff[oindex];
                }
            }
        }
    }

    #pragma omp parallel for
    for (int n = 0; n < in_chunk->num(); ++n) {
        for (int c = 0; c < in_chunk->channels(); ++c) {
            const int mindex = mean_->offset(n, c, 0, 0);
            for (int h = 0; h < in_chunk->height(); ++h) {
                for (int w = 0; w < in_chunk->width(); ++w) {
                    const int iindex = in_chunk->offset(n, c, h, w);
                    in_diff[iindex] += (out_no_shift_diff[iindex] / std::sqrt(var_data[mindex]+1e-8f)
                                        + 2 * var_diff[mindex] * (in_data[iindex]-mean_data[mindex]) / m
                                        + mean_diff[mindex] / m);
                }
            }
        }
    }
}

void InstanceNormalization::initialize() {
    float* gamma_data = params_[0]->data();
    float* beta_data = params_[1]->data();
    normal_random_init(params_[0]->count(), gamma_data, 0.0f, 0.1f);
    constant_init(params_[1]->count(), beta_data, 0.1f);
}

vector<int> InstanceNormalization::shape_inference() {
    return chunks_in_[0]->shape();
}

} // namespace micronet
