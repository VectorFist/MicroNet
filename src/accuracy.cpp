/**
 * @file accuracy.cpp
 * @auther yefajie
 * @data 2018/6/24
 **/
#include "accuracy.h"
#include <iostream>

namespace micronet {

Accuracy::Accuracy(const string& layer_name): Layer(layer_name, "Accuracy") {
    cout << "Initialize acc layer: " << layer_name_ << " done..." << endl;
}

chunk_ptr Accuracy::operator()(const chunk_ptr& in_chunk1, const chunk_ptr& in_chunk2) {
    chunks_in_ = {in_chunk1, in_chunk2};
    vector<int> out_shape = shape_inference();
    chunk_ptr out_chunk = make_shared<Chunk>(out_shape);
    chunks_out_ = {out_chunk};

    layer_ptr layer = make_shared<Accuracy>(*this);
    in_chunk1->in_layers_.push_back(layer);
    in_chunk2->in_layers_.push_back(layer);
    out_chunk->out_layer_ = layer;

    return out_chunk;
}

void Accuracy::forward(bool is_train) {
    Chunk* prob = chunks_in_[0].get();
    Chunk* labels = chunks_in_[1].get();
    if ((prob->count() / prob->channels()) != labels->count()) {
        cout << "Accuracy Error: labels shape must match prob shape(" << labels->str_shape() << "!=" << prob->str_shape() << ")";
        exit(1);
    }

    chunk_ptr predictions = make_shared<Chunk>();
    ArgMax argmax("argmax");
    argmax.chunks_in_ = {chunks_in_[0]};
    argmax.chunks_out_ = {predictions};
    argmax.forward();

    chunks_out_[0]->reshape(1, 1, 1, 1);
    const float* labels_data = labels->const_data();
    const float* predictions_data = predictions->const_data();
    float* output_data = chunks_out_[0]->data();

    int corr_count = 0;
    for (int i = 0; i < labels->count(); ++i) {
        if (labels_data[i] == predictions_data[i]) {
            corr_count++;
        }
    }
    output_data[0] = float(corr_count) / labels->count();
}

vector<int> Accuracy::shape_inference() {
    return {1, 1, 1, 1};
}

} // namespace micronet
