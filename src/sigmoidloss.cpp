#include <cmath>
#include <iostream>
#include <omp.h>
#include <string.h>
#include "sigmoidloss.h"

namespace micronet {

SigmoidLoss::SigmoidLoss(const string& layer_name): Layer(layer_name, "SigmoidLoss") {
    cout << "Initialize sigmoidloss layer: " << layer_name_ << " done..." << endl;
}

vector<chunk_ptr> SigmoidLoss::operator()(chunk_ptr& in_chunk1, chunk_ptr& in_chunk2) {
    if (in_chunk1->str_shape() != in_chunk2->str_shape()) {
        cout << "SigmoidLoss Error: labels shape must be same with logits shape(" <<
                in_chunk2->str_shape() << "!=" << in_chunk1->str_shape() << ")";
        exit(1);
    }

    chunks_in_ = {in_chunk1, in_chunk2};
    chunk_ptr out_chunk1 = make_shared<Chunk>(1, 1, 1, 1);
    chunk_ptr out_chunk2 = make_shared<Chunk>(in_chunk1->shape());
    chunks_out_ = {out_chunk1, out_chunk2};

    layer_ptr layer = make_shared<SigmoidLoss>(*this);
    in_chunk1->in_layers_.push_back(layer);
    in_chunk2->in_layers_.push_back(layer);
    out_chunk1->out_layer_ = layer;
    out_chunk2->out_layer_ = layer;

    return {out_chunk1, out_chunk2};
}


void SigmoidLoss::forward(bool is_train) {
    Chunk* logits = chunks_in_[0].get();
    Chunk* labels = chunks_in_[1].get();

    chunks_out_[0]->reshape(1, 1, 1, 1);
    chunks_out_[1]->reshape(logits->shape());

    const float* logits_data = logits->const_data();
    const float* labels_data = labels->const_data();
    float* prob_data = chunks_out_[1]->data();
    float* loss_data = chunks_out_[0]->data();
    float* loss_diff = chunks_out_[0]->diff();

    for (int i = 0; i < chunks_out_[1]->count(); ++i) {
        prob_data[i] = 1 / (1 + std::exp(-logits_data[i]));
    }
    float loss = 0;
    for (int i = 0; i < logits->count(); ++i) {
        //float tmp = labels_data[i]*std::log(max(prob_data[i], FLT_MIN)) +
        //            (1-labels_data[i])*std::log(max(1-prob_data[i], FLT_MIN));
        //loss -= tmp;
        loss -= logits_data[i] * (labels_data[i] - (logits_data[i] >= 0)) -
                log(1 + exp(logits_data[i] - 2 * logits_data[i] * (logits_data[i] >= 0)));
    }

    loss_data[0] = loss / labels->count();
    //loss_diff[0] = 1;
    gradient_reset();
}

void SigmoidLoss::backward() {
    Chunk* logits = chunks_in_[0].get();
    Chunk* labels = chunks_in_[1].get();

    const float* labels_data = labels->const_data();
    const float* prob_data = chunks_out_[1]->const_data();
    const float* loss_diff = chunks_out_[0]->const_diff();
    const float* logits_data = logits->const_data();
    float* logits_diff = logits->diff();
    //memcpy(logits_diff, prob_data, logits->count()*sizeof(float));
    for (int i = 0; i < logits->count(); ++i) {
        logits_diff[i] = prob_data[i] - labels_data[i];
    }

    for (int i = 0; i < logits->count(); ++i) {
        logits_diff[i] *= (loss_diff[0] / labels->count());
    }
}

vector<int> SigmoidLoss::shape_inference() {

}

} // namespace micronet
