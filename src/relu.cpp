/**
 * @file relu.cpp
 * @auther yefajie
 * @data 2018/6/23
 **/
#include <omp.h>
#include "relu.h"

namespace micronet {

ReLU::ReLU(const string& layer_name): Layer(layer_name, "ReLU") {
    cout << "Initialize relu layer: " << layer_name_ << " done..." << endl;
}

chunk_ptr ReLU::operator()(chunk_ptr& in_chunk) {
    chunk_ptr out_chunk = make_shared<Chunk>(in_chunk->shape());
    chunks_in_ = {in_chunk};
    chunks_out_ = {out_chunk};

    in_chunk->in_layers_.push_back(make_shared<ReLU>(*this));
    return out_chunk;
}
} // namespace micronet
