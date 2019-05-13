#include <cstring>
#include "omp.h"
#include "batchnormalization.h"
#include "math_func.h"
#include "util.h"

namespace micronet {

BatchNormalization::BatchNormalization(const string& layer_name): Layer(layer_name, "BatchNormalization"),
                    out_no_shift_(new Chunk()) {
    int_hps_["iter"] = 0;
}

chunk_ptr BatchNormalization::operator()(const chunk_ptr& in_chunk) {
    chunks_in_ = {in_chunk};
    chunk_ptr out_chunk = make_shared<Chunk>(shape_inference());
    chunks_out_ = {out_chunk};

    auto mean = make_shared<Chunk>(1, in_chunk->channels(), 1, 1);
    auto var = make_shared<Chunk>(1, in_chunk->channels(), 1, 1);
    auto running_avg_mean = make_shared<Chunk>(1, in_chunk->channels(), 1, 1);
    auto running_avg_var = make_shared<Chunk>(1, in_chunk->channels(), 1, 1);
    auto gamma = make_shared<Chunk>(1, in_chunk->channels(), 1, 1);
    auto beta = make_shared<Chunk>(1, in_chunk->channels(), 1, 1);
    mean->set_trainable(false);
    var->set_trainable(false);
    running_avg_mean->set_trainable(false);
    running_avg_var->set_trainable(false);
    params_ = {mean, var, running_avg_mean, running_avg_var, gamma, beta};

    initialize();

    layer_ptr layer = make_shared<BatchNormalization>(*this);
    in_chunk->in_layers_.push_back(layer);
    out_chunk->out_layer_ = layer;

    return out_chunk;
}

void BatchNormalization::forward(bool is_train) {
    auto in_chunk = chunks_in_[0];
    auto out_chunk = chunks_out_[0];
    out_chunk->reshape(shape_inference());
    out_no_shift_->reshape(shape_inference());

    auto mean = params_[0];
    auto var = params_[1];
    auto running_avg_mean = params_[2];
    auto running_avg_var = params_[3];
    auto gamma = params_[4];
    auto beta = params_[5];

    const float* in_data = in_chunk->const_data();
    const float* gamma_data = gamma->const_data();
    const float* beta_data = beta->const_data();
    float* out_no_shift_data = out_no_shift_->data();
    float* out_data = out_chunk->data();
    float* mean_data = mean->data();
    float* var_data = var->data();
    float* avg_mean_data = running_avg_mean->data();
    float* avg_var_data = running_avg_var->data();

    if (is_train) {
        const int m = in_chunk->count() / in_chunk->channels();
        #pragma omp parallel for
        for (int c = 0; c < in_chunk->channels(); ++c) {
            const int mindex = mean->offset(0, c, 0, 0);
            mean_data[mindex] = 0;
            for (int n = 0; n < in_chunk->num(); ++n) {
                for (int h = 0; h < in_chunk->height(); ++h) {
                    for (int w = 0; w < in_chunk->width(); ++w) {
                        const int iindex = in_chunk->offset(n, c, h, w);
                        mean_data[mindex] += in_data[iindex];
                    }
                }
            }
            mean_data[mindex] /= m;
        }
        #pragma omp parallel for
        for (int c = 0; c < in_chunk->channels(); ++c) {
            const int vindex = var->offset(0, c, 0, 0);
            var_data[vindex] = 0;
            for (int n = 0; n < in_chunk->num(); ++n) {
                for (int h = 0; h < in_chunk->height(); ++h) {
                    for (int w = 0; w < in_chunk->width(); ++w) {
                        const int iindex = in_chunk->offset(n, c, h, w);
                        var_data[vindex] += std::pow(in_data[iindex] - mean_data[vindex], 2);
                    }
                }
            }
            var_data[vindex] /= m;
        }

        if (int_hps_["iter"] == 0) {
            memcpy(avg_mean_data, mean_data, mean->count()*sizeof(float));
            memcpy(avg_var_data, var_data, var->count()*sizeof(float));
        } else {
            add(mean->count(), avg_mean_data, 0.1, mean_data, 0.9, avg_mean_data);
            add(var->count(), avg_var_data, 0.1, var_data, 0.9, avg_var_data);
        }
        int_hps_["iter"]++;

        #pragma omp parallel for
        for (int n = 0; n < in_chunk->num(); ++n) {
            for (int c = 0; c < in_chunk->channels(); ++c) {
                const int mindex = mean->offset(0, c, 0, 0);
                for (int h = 0; h < in_chunk->height(); ++h) {
                    for (int w = 0; w < in_chunk->width(); ++w) {
                        const int iindex = in_chunk->offset(n, c, h, w);
                        out_no_shift_data[iindex] = (in_data[iindex] - mean_data[mindex])
                                            / std::sqrt(var_data[mindex] + 1e-8f);
                        out_data[iindex] = gamma_data[mindex] * out_no_shift_data[iindex] + beta_data[mindex];
                    }
                }
            }
        }
    } else {
        #pragma omp parallel for
        for (int n = 0; n < in_chunk->num(); ++n) {
            for (int c = 0; c < in_chunk->channels(); ++c) {
                const int mindex = running_avg_mean->offset(0, c, 0, 0);
                for (int h = 0; h < in_chunk->height(); ++h) {
                    for (int w = 0; w < in_chunk->width(); ++w) {
                        const int iindex = in_chunk->offset(n, c, h, w);
                        out_no_shift_data[iindex] = (in_data[iindex] - avg_mean_data[mindex])
                                            / std::sqrt(avg_var_data[mindex] + 1e-8f);
                        out_data[iindex] = gamma_data[mindex] * out_no_shift_data[iindex] + beta_data[mindex];
                    }
                }
            }
        }
    }

    gradient_reset();
}

void BatchNormalization::backward() {
    auto in_chunk = chunks_in_[0];
    auto out_chunk = chunks_out_[0];
    auto mean = params_[0];
    auto var = params_[1];
    auto gamma = params_[4];
    auto beta = params_[5];

    const float* in_data = in_chunk->const_data();
    const float* mean_data = mean->const_data();
    const float* var_data = var->const_data();
    const float* gamma_data = gamma->const_data();
    const float* out_diff = out_chunk->const_diff();
    const float* out_no_shift_data = out_no_shift_->const_data();
    float* out_no_shift_diff = out_no_shift_->diff();
    float* in_diff = in_chunk->diff();
    float* mean_diff = mean->diff();
    float* var_diff = var->diff();
    float* gamma_diff = gamma->diff();
    float* beta_diff = beta->diff();

    const int m = in_chunk->count() / in_chunk->channels();
    #pragma omp parallel for
    for(int c = 0; c < in_chunk->channels(); ++c) {
        float tmp0 = 0;
        for (int n = 0; n < in_chunk->num(); ++n) {
            for (int h = 0; h < in_chunk->height(); ++h) {
                for (int w = 0; w < in_chunk->width(); ++w) {
                    const int oindex = out_chunk->offset(n, c, h, w);
                    out_no_shift_diff[oindex] = gamma_data[c] * out_diff[oindex];
                    var_diff[c] += out_no_shift_diff[oindex] * (in_data[oindex]-mean_data[c])
                                   * (-0.5) * std::pow(var_data[c]+1e-8f, -1.5);
                    mean_diff[c] += (-out_no_shift_diff[oindex]) / std::sqrt(var_data[c]+1e-8f);
                    tmp0 += 2 * (mean_data[c] - in_data[oindex]);

                    gamma_diff[c] += out_diff[oindex] * out_no_shift_data[oindex];
                    beta_diff[c] += out_diff[oindex];
                }
            }
        }
        mean_diff[c] += var_diff[c] * tmp0 / m;
    }

    #pragma omp parallel for
    for (int n = 0; n < in_chunk->num(); ++n) {
        for (int c = 0; c < in_chunk->channels(); ++c) {
            for (int h = 0; h < in_chunk->height(); ++h) {
                for (int w = 0; w < in_chunk->width(); ++w) {
                    const int iindex = in_chunk->offset(n, c, h, w);
                    in_diff[iindex] += (out_no_shift_diff[iindex] / std::sqrt(var_data[c]+1e-8f)
                                        + 2 * var_diff[c] * (in_data[iindex]-mean_data[c]) / m
                                        + mean_diff[c] / m);
                }
            }
        }
    }
}

void BatchNormalization::initialize() {
    float* gamma_data = params_[4]->data();
    float* beta_data = params_[5]->data();
    normal_random_init(params_[4]->count(), gamma_data, 0.0f, 0.1f);
    constant_init(params_[5]->count(), beta_data, 0.1f);
}

vector<int> BatchNormalization::shape_inference() {
    return chunks_in_[0]->shape();
}

} // namespace micronet
