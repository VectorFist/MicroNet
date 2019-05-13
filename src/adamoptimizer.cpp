/**
 * @file adamoptimizer.cpp
 * @auther yefajie
 * @data 2018/6/25
 **/
#include <thread>
#include <omp.h>
#include "adamoptimizer.h"

namespace micronet {

AdamOptimizer::AdamOptimizer(float learning_rate, vector<float> decay_locs, float decay_rate_1, float decay_rate_2):
     Optimizer("Adam", decay_locs) {
    flt_hps_["learning_rate"] = learning_rate;
    flt_hps_["decay_rate_1"] = decay_rate_1;
    flt_hps_["decay_rate_2"] = decay_rate_2;
    flt_hps_["decay_rate_1_pow"] = 1;
    flt_hps_["decay_rate_2_pow"] = 1;
    int_hps_["iter"] = 0;
}

void AdamOptimizer::optimize(const shared_ptr<Chunk>& param, int iter) {
    float decay_loc = (float)iter / total_iters_;
    int decay_index = find_if(decay_locs_.begin(), decay_locs_.end(),
                              [&decay_loc](float loc) {return loc > decay_loc;}) - decay_locs_.begin();
    float learning_rate = flt_hps_["learning_rate"] * pow(0.1, decay_index);

    if (iter != int_hps_["iter"]) {
        //cout << decay_rate_1_pow_ << endl;
        //exit(0);
        int_hps_["iter"] = iter;
        flt_hps_["decay_rate_1_pow"] *= flt_hps_["decay_rate_1"];
        flt_hps_["decay_rate_2_pow"] *= flt_hps_["decay_rate_2"];
        /*if (iter %100 == 0) {
            cout << "iter:" << iter << endl;
            cout << decay_rate_1_ << " " << decay_rate_2_ << endl;
            cout << "decay1:" << decay_rate_1_pow_ << endl;
            cout << "decay2:" << decay_rate_2_pow_ << endl;
        }*/
    }

    if (first_moment_estimate_.find(param.get()) == first_moment_estimate_.end()) {
        first_moment_estimate_[param.get()] = Chunk(param->shape());
    }
    if (second_moment_estimate_.find(param.get()) == second_moment_estimate_.end()) {
        second_moment_estimate_[param.get()] = Chunk(param->shape());
    }
    const float* diff = param->const_diff();
    float* data = param->data();
    float* first_moment_est = first_moment_estimate_[param.get()].data();
    float* second_moment_est = second_moment_estimate_[param.get()].data();

    float decay_rate_1 = flt_hps_["decay_rate_1"];
    float decay_rate_2 = flt_hps_["decay_rate_2"];
    float decay_rate_1_pow = flt_hps_["decay_rate_1_pow"];
    float decay_rate_2_pow = flt_hps_["decay_rate_2_pow"];

    #pragma omp parallel for
    for (int i = 0; i < param->count(); i++) {
        first_moment_est[i] = decay_rate_1 * first_moment_est[i] + (1 - decay_rate_1) * diff[i];
        second_moment_est[i] = decay_rate_2 * second_moment_est[i] + (1 - decay_rate_2) * diff[i] * diff[i];
        //float corrected_first_moment = first_moment_est[i] / (1 - decay_rate_1_pow_);
        //float corrected_second_moment = second_moment_est[i] / (1 - decay_rate_2_pow_);
        //float update = (-learning_rate) * corrected_first_moment / (std::sqrt(corrected_second_moment) + 1e-8F);
        float alpha_t = learning_rate * std::sqrt(1-decay_rate_2_pow) / (1-decay_rate_1_pow);
        float update = (-alpha_t) * first_moment_est[i] / (std::sqrt(second_moment_est[i])+1e-8f);
        data[i] += update;
    }
}

} // namespace micronet
