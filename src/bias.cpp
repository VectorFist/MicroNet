/**
 * @file bias.cpp
 * @auther yefajie
 * @data 2018/6/23
 **/
#include <iostream>
#include "bias.h"

namespace micronet {

Bias::Bias(int input_channels, const string& layer_name, float bias_value): Layer(layer_name, "Bias") {
    params_.push_back(make_shared<Chunk>(1, input_channels, 1, 1));
    cout << "Initialize bias layer: " << layer_name_ << " done..." << endl;
}

} // namespace micronet
