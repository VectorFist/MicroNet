#include <string.h>

#include "add.h"
#include "util.h"
#include "math_func.h"

namespace micronet {

Add::Add(const string& layer_name): Layer(layer_name, "Add") {
    cout << "Initialize add layer: " << layer_name_ << " done..." << endl;
}

chunk_ptr Add::operator()(const chunk_ptr& in_chunk1, const chunk_ptr& in_chunk2) {
    if (in_chunk1->str_shape() != in_chunk2->str_shape()) {
        cout << "Add error, chunks must have same shape: " << in_chunk1->str_shape() <<
                " != " << in_chunk2->str_shape() << endl;
    }

    chunks_in_ = {in_chunk1, in_chunk2};
    chunk_ptr out_chunk = make_shared<Chunk>(shape_inference());
    chunks_out_ = {out_chunk};

    layer_ptr layer = make_shared<Add>(*this);
    in_chunk1->in_layers_.push_back(layer);
    in_chunk2->in_layers_.push_back(layer);
    out_chunk->out_layer_ = layer;

    return out_chunk;
}

void Add::forward(bool is_train) {
    Timer timer;
    chunks_out_[0]->reshape(shape_inference());

    const float* input1_data = chunks_in_[0]->const_data();
    const float* input2_data = chunks_in_[1]->const_data();
    float* output_data = chunks_out_[0]->data();

    add(chunks_out_[0]->count(), input1_data, 1, input2_data, 1, output_data);
    //cout << "add forward layer: " << timer.elapsed()*1000 << endl;
    //exit(0);
    gradient_reset();
}

void Add::backward() {
    Timer timer;
    float* input1_diff = chunks_in_[0]->diff();
    float* input2_diff = chunks_in_[1]->diff();
    const float*  output_diff = chunks_out_[0]->const_diff();

    add(chunks_out_[0]->count(), input1_diff, 1, output_diff, 1, input1_diff);
    add(chunks_out_[0]->count(), input2_diff, 1, output_diff, 1, input2_diff);
    //cout << "add backward layer: " << timer.elapsed()*1000 << endl;
    //exit(0);
}

vector<int> Add::shape_inference() {
    return chunks_in_[0]->shape();
}

} // namespace micronet
