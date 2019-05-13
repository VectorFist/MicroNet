/**
 * @file adagradoptimizer.cpp
 * @auther yefajie
 * @data 2018/6/25
 **/
#include <cmath>
#include "adagradoptimizer.h"

namespace micronet {

AdaGradOptimizer::AdaGradOptimizer(float learning_rate, vector<float> decay_locs):
    Optimizer("AdaGrad", decay_locs) {
    flt_hps_["learning_rate"] = learning_rate;
}

void AdaGradOptimizer::optimize(const shared_ptr<Chunk>& param, int iter) {
    float decay_loc = (float)iter / total_iters_;
    int decay_index = find_if(decay_locs_.begin(), decay_locs_.end(),
                              [&decay_loc](float loc) {return loc > decay_loc;}) - decay_locs_.begin();
    float learning_rate = flt_hps_["learning_rate"] * pow(0.1, decay_index);

    if (accumulate_squared_gradient_.find(param.get()) == accumulate_squared_gradient_.end()) {
        accumulate_squared_gradient_[param.get()] = Chunk(param->shape());
    }
    const float* diff = param->const_diff();
    float* data = param->data();
    float* acc_squared_grad = accumulate_squared_gradient_[param.get()].data();
    for (int i = 0; i < param->count(); ++i) {
        acc_squared_grad[i]  += diff[i] * diff[i];
        float update = -(learning_rate / sqrt(acc_squared_grad[i] + 1e-7F)) * diff[i];
        data[i] += update;
    }
}

} // namespace micronet
