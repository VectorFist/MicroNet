#include <cstring>
#include "dropout.h"
#include "util.h"

namespace micronet {

Dropout::Dropout(float keep_prob, const string& layer_name): Layer(layer_name, "Dropout"), mask_(new Chunk()) {
    flt_hps_["keep_prob"] = keep_prob;
}

chunk_ptr Dropout::operator()(const chunk_ptr& in_chunk) {
    chunks_in_ = {in_chunk};
    chunk_ptr out_chunk = make_shared<Chunk>(shape_inference());
    chunks_out_ = {out_chunk};

    layer_ptr layer = make_shared<Dropout>(*this);
    in_chunk->in_layers_.push_back(layer);
    out_chunk->out_layer_ = layer;

    return out_chunk;
}

void Dropout::forward(bool is_train) {
    chunk_ptr in_chunk = chunks_in_[0];
    chunk_ptr out_chunk = chunks_out_[0];
    float keep_prob = flt_hps_["keep_prob"];
    out_chunk->reshape(shape_inference());

    const float* in_data = in_chunk->const_data();
    float* out_data = out_chunk->data();
    memset(out_data, 0, out_chunk->count()*sizeof(float));

    if (is_train) {
        mask_->reshape(in_chunk->shape());
        float* mask_data = mask_->data();
        memset(mask_data, 0, mask_->count()*sizeof(float));

        UniformGenerator generator(0.0f, 1.0f);
        for (int i = 0; i < in_chunk->count(); ++i) {
            float prob = generator();
            if (prob < keep_prob) {
                out_data[i] = in_data[i] / keep_prob;
                mask_data[i] = 1;
            }
        }
    } else {
        memcpy(out_data, in_data, out_chunk->count()*sizeof(float));
    }

    gradient_reset();
}

void Dropout::backward() {
    chunk_ptr in_chunk = chunks_in_[0];
    chunk_ptr out_chunk = chunks_out_[0];
    float keep_prob = flt_hps_["keep_prob"];

    float* in_diff = in_chunk->diff();
    const float* out_diff = out_chunk->const_diff();
    const float* mask_data = mask_->const_data();

    for (int i = 0; i < in_chunk->count(); ++i) {
        if (mask_data[i]) {
            in_diff[i] += out_diff[i] / keep_prob;
        }
    }
}

vector<int> Dropout::shape_inference() {
    return chunks_in_[0]->shape();
}

} // namespace micronet
