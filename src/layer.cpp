/**
 * @file layer.cpp
 * @auther yefajie
 * @data 2018/6/22
 **/
 #include <string.h>

#include "layer.h"

namespace micronet {

Layer::Layer(const string& layer_name, const string& layer_type):
    layer_name_(layer_name), layer_type_(layer_type) {
}

void Layer::gradient_reset() {
    for (chunk_ptr& chunk: chunks_in_) {
        float* diff = chunk->diff();
        memset(diff, 0, chunk->count()*sizeof(float));
    }
    for (chunk_ptr& param: params_) {
        float* diff = param->diff();
        memset(diff, 0, param->count()*sizeof(float));
    }
}

} // namespace micronet
