/**
 * @file sgdoptimizer.cpp
 * @auther yefajie
 * @data 2018/6/25
 **/
#include "sgdoptimizer.h"

SGDOptimizer::SGDOptimizer(float learning_rate, const vector<int>& decay_steps, float momentum):
    Optimizer(learning_rate, decay_steps), momentum_(momentum) {
}

void SGDOptimizer::optimize(shared_ptr<Chunk>& param, const string& param_name, int iter) {
    int decay_index = find_if(decay_steps_.begin(), decay_steps_.end(),
                              [&iter](int step) {return step > iter;}) - decay_steps_.begin();
    float learning_rate = learning_rate_ * pow(0.1, decay_index);

    if (param_velocity_.find(param_name) == param_velocity_.end()) {
        param_velocity_[param_name] = Chunk(param->shape());
    }
    const float* diff = param->const_diff();
    float* data = param->data();
    float* velocity = param_velocity_[param_name].data();
    for (int i = 0; i < param->count(); ++i) {
        velocity[i]  = momentum_ * velocity[i] - learning_rate * diff[i];
        data[i] += velocity[i];
    }
}
