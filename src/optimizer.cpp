/**
 * @file optimizer.cpp
 * @auther yefajie
 * @data 2018/6/25
 **/
#include "optimizer.h"

namespace micronet {

Optimizer::Optimizer(const string& optimizer_type, vector<float> decay_locs):
    optimizer_type_(optimizer_type), decay_locs_(decay_locs) {
}

} // namespace micronet
