/**
 * @file adamoptimizer.cpp
 * @auther yefajie
 * @data 2018/6/25
 **/
#include "adamoptimizer.h"

AdamOptimizer::AdamOptimizer(float learning_rate, const vector<int>& decay_steps, float decay_rate_1, float decay_rate_2):
     Optimizer(learning_rate, decay_steps), decay_rate_1_(decay_rate_1), decay_rate_2_(decay_rate_2) {
}

void AdamOptimizer::optimize(shared_ptr<Chunk>& param, const string& param_name, int iter) {
    int decay_index = find_if(decay_steps_.begin(), decay_steps_.end(),
                              [&iter](int step) {return step > iter;}) - decay_steps_.begin();
    float learning_rate = learning_rate_ * pow(0.1, decay_index);

    if (first_moment_estimate_.find(param_name) == first_moment_estimate_.end()) {
        first_moment_estimate_[param_name] = Chunk(param->shape());
    }
    if (second_moment_estimate_.find(param_name) == second_moment_estimate_.end()) {
        second_moment_estimate_[param_name] = Chunk(param->shape());
    }
    const float* diff = param->const_diff();
    float* data = param->data();
    float* first_moment_est = first_moment_estimate_[param_name].data();
    float* second_moment_est = second_moment_estimate_[param_name].data();

    for (int i = 0; i < param->count(); i++) {
        first_moment_est[i] = decay_rate_1_ * first_moment_est[i] + (1 - decay_rate_1_) * diff[i];
        second_moment_est[i] = decay_rate_2_ * second_moment_est[i] + (1 - decay_rate_2_) * diff[i] * diff[i];
        float corrected_first_moment = first_moment_est[i] / (1 - std::pow(decay_rate_1_, iter));
        float corrected_second_moment = second_moment_est[i] / (1 - std::pow(decay_rate_2_, iter));
        float update = (-learning_rate) * (corrected_first_moment / (std::sqrt(corrected_second_moment) + 1e-8F));
        data[i] += update;
    }
}
