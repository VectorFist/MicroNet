/**
 * @file sgdoptimizer.cpp
 * @auther yefajie
 * @data 2018/6/25
 **/
#include <thread>
#include "sgdoptimizer.h"

namespace micronet {

SGDOptimizer::SGDOptimizer(float learning_rate, vector<float> decay_locs, float momentum):
    Optimizer("SGD", decay_locs) {
    flt_hps_["learning_rate"] = learning_rate;
    flt_hps_["momentum"] = momentum;
}

void SGDOptimizer::optimize(const shared_ptr<Chunk>& param, int iter) {
    float decay_loc = (float)iter / total_iters_;
    int decay_index = find_if(decay_locs_.begin(), decay_locs_.end(),
                              [&decay_loc](float loc) {return loc > decay_loc;}) - decay_locs_.begin();
    float learning_rate = flt_hps_["learning_rate"] * pow(0.1, decay_index);
    float momentum = flt_hps_["momentum"];

    if (param_velocity_.find(param.get()) == param_velocity_.end()) {
        param_velocity_[param.get()] = Chunk(param->shape());
    }
    const float* diff = param->const_diff();
    float* data = param->data();
    float* velocity = param_velocity_[param.get()].data();

    for (int i = 0; i < param->count(); ++i) {
        velocity[i]  = - momentum * velocity[i] - learning_rate * diff[i];
        data[i] += velocity[i];
    }
}

} // namespace micronet
