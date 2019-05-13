#include <omp.h>
#include <string.h>
#include "batchmiddlesplit.h"
#include "math_func.h"

namespace micronet {

BatchMiddleSplit::BatchMiddleSplit(const string& layer_name):
    Layer(layer_name, "BatchMiddleSplit") {
}

pair<chunk_ptr, chunk_ptr> BatchMiddleSplit::operator()(const chunk_ptr& in_chunk) {
    if (in_chunk->num() % 2 != 0) {
        cout << "inchunk num must be odd!";
        exit(1);
    }

    chunks_in_ = {in_chunk};
    vector<int> out_shape = shape_inference();
    chunk_ptr out_chunk1 = make_shared<Chunk>(out_shape);
    chunk_ptr out_chunk2 = make_shared<Chunk>(out_shape);
    chunks_out_ = {out_chunk1, out_chunk2};

    layer_ptr layer = make_shared<BatchMiddleSplit>(*this);
    in_chunk->in_layers_.push_back(layer);
    out_chunk1->out_layer_ = layer;
    out_chunk2->out_layer_ = layer;

    return make_pair(out_chunk1, out_chunk2);
}

void BatchMiddleSplit::forward(bool is_train) {
    if (chunks_in_[0]->num() % 2 != 0) {
        cout << "inchunk num must be odd!";
        exit(1);
    }
    vector<int> out_shape = shape_inference();
    auto out_chunk1 = chunks_out_[0];
    auto out_chunk2 = chunks_out_[1];
    out_chunk1->reshape(out_shape);
    out_chunk2->reshape(out_shape);

    const float* in_data = chunks_in_[0]->const_data();
    float* out_data1 = out_chunk1->data();
    float* out_data2 = out_chunk2->data();

    int half_count = chunks_in_[0]->count() / 2;
    memcpy(out_data1, in_data, half_count*sizeof(float));
    memcpy(out_data2, in_data+half_count, half_count*sizeof(float));

    gradient_reset();
}

void BatchMiddleSplit::backward() {
    auto out_chunk1 = chunks_out_[0];
    auto out_chunk2 = chunks_out_[1];

    float* in_diff = chunks_in_[0]->diff();
    const float* out_diff1 = out_chunk1->const_diff();
    const float* out_diff2 = out_chunk2->const_diff();

    int half_count = chunks_in_[0]->count() / 2;
    add(half_count, in_diff, 1.0f, out_diff1, 1.0, in_diff);
    add(half_count, in_diff+half_count, 1.0f, out_diff2, 1.0, in_diff+half_count);
}

vector<int> BatchMiddleSplit::shape_inference() {
    return {chunks_in_[0]->shape(0)/2, chunks_in_[0]->shape(1), chunks_in_[0]->shape(2), chunks_in_[0]->shape(3)};
}

} // namespace micronet

