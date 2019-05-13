#include <algorithm>
#include <cmath>
#include "activation.h"
#include "util.h"

namespace micronet {

Activation::Activation(const string& activation, float leaky_alpha, const string& layer_name):
    Layer(layer_name, "Activation") {

    str_hps_["activation"] = activation;
    flt_hps_["leaky_alpha"] = leaky_alpha;
    flt_hps_["selu_lambda"] = 1.05070099f;
    flt_hps_["selu_alpha"] = 1.67326324f;

    if (all_activations_.find(activation) == all_activations_.end()) {
        cout << "Activation: " << activation << " is not implemented..." << endl;
        exit(1);
    }
    cout << "Initialize activation layer: " << layer_name << " done..." << endl;
}

chunk_ptr Activation::operator()(const chunk_ptr& in_chunk) {
    chunks_in_ = {in_chunk};
    chunk_ptr out_chunk = make_shared<Chunk>(shape_inference());
    chunks_out_ = {out_chunk};

    if (str_hps_["activation"] == "prelu") {
        params_.push_back(make_shared<Chunk>(1, in_chunk->channels(), 1, 1));
        constant_init(params_[0]->count(), params_[0]->data(), 0.0f);
    }

    layer_ptr layer = make_shared<Activation>(*this);
    in_chunk->in_layers_.push_back(layer);
    out_chunk->out_layer_ = layer;

    return out_chunk;
}

void Activation::forward(bool is_train) {
    chunks_out_[0]->reshape(shape_inference());
    const float* input_data = chunks_in_[0]->const_data();
    float* output_data = chunks_out_[0]->data();

    if (str_hps_["activation"] == "relu") {
        #pragma omp parallel for
        for (int i = 0; i < chunks_in_[0]->count(); ++i) {
            output_data[i] = std::max(0.0f, input_data[i]);
        }
    } else if (str_hps_["activation"] == "leaky_relu") {
        float leaky_alpha = flt_hps_["leaky_alpha"];
        #pragma omp parallel for
        for (int i = 0; i < chunks_in_[0]->count(); ++i) {
            output_data[i] = std::max(leaky_alpha*input_data[i], input_data[i]);
        }
    } else if (str_hps_["activation"] == "relu6") {
        #pragma omp parallel for
        for (int i = 0; i < chunks_in_[0]->count(); ++i) {
            output_data[i] = std::min(std::max(0.0f, input_data[i]), 6.0f);
        }
    } else if (str_hps_["activation"] == "sigmoid") {
        #pragma omp parallel for
        for (int i = 0; i < chunks_in_[0]->count(); ++i) {
            output_data[i] = 1 / (1 + std::exp(-input_data[i]));
        }
    } else if (str_hps_["activation"] == "tanh") {
        #pragma omp parallel for
        for (int i = 0; i < chunks_in_[0]->count(); ++i) {
            float pos_exp = std::exp(input_data[i]);
            float neg_exp = std::exp(-input_data[i]);
            output_data[i] = (pos_exp - neg_exp) / (pos_exp + neg_exp);
        }
    } else if (str_hps_["activation"] == "elu") {
        #pragma omp parallel for
        for (int i = 0; i < chunks_in_[0]->count(); ++i) {
            output_data[i] = input_data[i] > 0? input_data[i]: (std::exp(input_data[i]) - 1);
        }
    } else if (str_hps_["activation"] == "selu") {
        float selu_lambda = flt_hps_["selu_lambda"];
        float selu_alpha = flt_hps_["selu_alpha"];
        #pragma omp parallel for
        for (int i = 0; i < chunks_in_[0]->count(); ++i) {
            output_data[i] = selu_lambda * (input_data[i] > 0? input_data[i]: selu_alpha* (std::exp(input_data[i]) - 1));
        }
    } else if (str_hps_["activation"] == "prelu") {
        const float* alpha_data = params_[0]->const_data();
        chunk_ptr in_chunk = chunks_in_[0];
        chunk_ptr out_chunk = chunks_out_[0];
        #pragma omp parallel for
        for (int n = 0; n < in_chunk->num(); ++n) {
            for (int c = 0; c < in_chunk->channels(); ++c) {
                const int aindex = params_[0]->offset(0, c, 0, 0);
                float alpha = alpha_data[aindex];
                for (int h = 0; h < in_chunk->height(); ++h) {
                    for (int w = 0; w < in_chunk->width(); ++w) {
                        const int oindex = out_chunk->offset(n, c, h, w);
                        output_data[oindex] = input_data[oindex] > 0? input_data[oindex]: alpha * input_data[oindex];
                    }
                }
            }
        }
    } else if (str_hps_["activation"] == "sin") {
        #pragma omp parallel for
        for (int i = 0; i < chunks_in_[0]->count(); ++i) {
            output_data[i] = std::sin(input_data[i]);
        }
    }
    gradient_reset();
    //cout << "activation forward" << endl;
}

void Activation::backward() {
    const float* input_data = chunks_in_[0]->const_data();
    const float* output_data= chunks_out_[0]->const_data();
    const float* output_diff = chunks_out_[0]->const_diff();
    float* input_diff = chunks_in_[0]->diff();

    if (str_hps_["activation"] == "relu") {
        #pragma omp parallel for
        for (int i = 0; i < chunks_in_[0]->count(); ++i) {
            input_diff[i] += (input_data[i] > 0) * output_diff[i];
        }
    } else if (str_hps_["activation"] == "leaky_relu") {
        float leaky_alpha = flt_hps_["leaky_alpha"];
        #pragma omp parallel for
        for (int i = 0; i < chunks_in_[0]->count(); ++i) {
            input_diff[i] += input_data[i] > 0 ? output_diff[i]: leaky_alpha * output_diff[i];
        }
    } else if (str_hps_["activation"] == "relu6") {
        #pragma omp parallel for
        for (int i = 0; i < chunks_in_[0]->count(); ++i) {
            input_diff[i] += (input_data[i] > 0 && input_data[i] < 6) * output_diff[i];
        }
    } else if (str_hps_["activation"] == "sigmoid") {
        #pragma omp parallel for
        for (int i = 0; i < chunks_in_[0]->count(); ++i) {
            input_diff[i] += output_data[i] * (1-output_data[i]) * output_diff[i];
        }
    } else if (str_hps_["activation"] == "tanh") {
        #pragma omp parallel for
        for (int i = 0; i < chunks_in_[0]->count(); ++i) {
            input_diff[i] += (1 - std::pow(output_data[i], 2)) * output_diff[i];
        }
    } else if (str_hps_["activation"] == "elu") {
        #pragma omp parallel for
        for (int i = 0; i < chunks_in_[0]->count(); ++i) {
            input_diff[i] += input_data[i] > 0 ? output_diff[i]: std::exp(input_data[i]) * output_diff[i];
        }
    } else if (str_hps_["activation"] == "selu") {
        float selu_lambda = flt_hps_["selu_lambda"];
        float selu_alpha = flt_hps_["selu_alpha"];
        #pragma omp parallel for
        for (int i = 0; i < chunks_in_[0]->count(); ++i) {
            input_diff[i] += selu_lambda * (input_data[i] > 0 ? output_diff[i]: selu_alpha * std::exp(input_data[i])) * output_diff[i];
        }
    } else if (str_hps_["activation"] == "prelu") {
        const float* alpha_data = params_[0]->const_data();
        float* alpha_diff = params_[0]->diff();
        chunk_ptr in_chunk = chunks_in_[0];
        chunk_ptr out_chunk = chunks_out_[0];
        for (int n = 0; n < in_chunk->num(); ++n) {
            //#pragma omp parallel for
            for (int c = 0; c < in_chunk->channels(); ++c) {
                const int aindex = params_[0]->offset(0, c, 0, 0);
                float alpha = alpha_data[aindex];
                for (int h = 0; h < in_chunk->height(); ++h) {
                    for (int w = 0; w < in_chunk->width(); ++w) {
                        const int oindex = out_chunk->offset(n, c, h, w);
                        input_diff[oindex] += input_data[oindex] > 0? output_diff[oindex]: alpha * output_diff[oindex];
                        alpha_diff[aindex] += (input_data[oindex] < 0) * input_data[oindex] * output_diff[oindex];
                    }
                }
            }
        }
    } else if (str_hps_["activation"] == "sin") {
        #pragma omp parallel for
        for (int i = 0; i < chunks_in_[0]->count(); ++i) {
            input_diff[i] += std::cos(input_data[i]) * output_diff[i];
        }
    }
}

vector<int> Activation::shape_inference() {
    return chunks_in_[0]->shape();
}

} // namespace micronet
