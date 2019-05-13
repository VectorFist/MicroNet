#include "l2loss.h"

namespace micronet {

L2Loss::L2Loss(const string& layer_name): Layer(layer_name, "L2Loss") {
    //ctor
}

chunk_ptr L2Loss::operator()(const chunk_ptr& in_chunk1, const chunk_ptr& in_chunk2) {
    if (in_chunk1->str_shape() != in_chunk2->str_shape()) {
        cout << "input chunks must have same shape!";
        exit(1);
    }

    chunks_in_ = {in_chunk1, in_chunk2};
    chunk_ptr out_chunk = make_shared<Chunk>(shape_inference());
    chunks_out_ = {out_chunk};

    layer_ptr layer = make_shared<L2Loss>(*this);
    in_chunk1->in_layers_.push_back(layer);
    in_chunk2->in_layers_.push_back(layer);
    out_chunk->out_layer_ = layer;

    return out_chunk;
}

void L2Loss::forward(bool is_train) {
    chunks_out_[0]->reshape(shape_inference());

    const float* pred_data = chunks_in_[0]->const_data();
    const float* target_data = chunks_in_[1]->const_data();
    float* loss_data = chunks_out_[0]->data();
    float* loss_diff = chunks_out_[0]->diff();

    float loss = 0;
    for (int i = 0; i < chunks_in_[0]->count(); ++i) {
        loss += std::pow(pred_data[i]-target_data[i], 2);
    }
    loss_data[0] = loss / chunks_in_[0]->count();
    loss_diff[0] = 1.0f;

    gradient_reset();
}

void L2Loss::backward() {
    float* pred_diff = chunks_in_[0]->diff();
    float* target_diff = chunks_in_[1]->diff();
    const float* loss_diff = chunks_out_[0]->const_diff();
    const float* pred_data = chunks_in_[0]->const_data();
    const float* target_data = chunks_in_[1]->const_data();

    int num_count = chunks_in_[0]->count();
    for (int i = 0; i < num_count; ++i) {
        pred_diff[i] += 2 * loss_diff[0] * (pred_data[i] - target_data[i]) / num_count;
        target_diff[i] += 2 * loss_diff[0] * (target_data[i] - pred_data[i]) / num_count;
    }
}

vector<int> L2Loss::shape_inference() {
    return {1, 1, 1, 1};
}

} // namespace micronet
