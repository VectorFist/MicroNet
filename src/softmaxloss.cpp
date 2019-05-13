/**
 * @file softmaxloss.cpp
 * @auther yefajie
 * @data 2018/6/24
 **/
#include <cmath>
#include <iostream>
#include <omp.h>
#include <string.h>
#include "softmaxloss.h"

namespace micronet {

SoftmaxLoss::SoftmaxLoss(const string& layer_name): Layer(layer_name, "SoftmaxLoss"), softmax_("softmax"), prob_(new Chunk()) {
    cout << "Initialize softmaxloss layer: " << layer_name_ << " done..." << endl;
}

vector<chunk_ptr> SoftmaxLoss::operator()(chunk_ptr& in_chunk1, chunk_ptr& in_chunk2) {
    if ((in_chunk1->count() / in_chunk1->channels()) != in_chunk2->count()) {
        cout << "SoftmaxLoss Error: labels shape must match logits shape(" <<
                in_chunk2->str_shape() << "!=" << in_chunk1->str_shape() << ")";
        exit(1);
    }

    chunks_in_ = {in_chunk1, in_chunk2};
    chunk_ptr out_chunk1 = make_shared<Chunk>(1, 1, 1, 1);
    chunk_ptr out_chunk2 = make_shared<Chunk>(in_chunk1->shape());
    chunks_out_ = {out_chunk1, out_chunk2};

    softmax_.chunks_in_ = {in_chunk1};
    softmax_.chunks_out_ = {prob_};

    layer_ptr layer = make_shared<SoftmaxLoss>(*this);
    in_chunk1->in_layers_.push_back(layer);
    in_chunk2->in_layers_.push_back(layer);
    out_chunk1->out_layer_ = layer;
    out_chunk2->out_layer_ = layer;

    return {out_chunk1, out_chunk2};
}


void SoftmaxLoss::forward(bool is_train) {
    Chunk* logits = chunks_in_[0].get();
    Chunk* labels = chunks_in_[1].get();

    if (softmax_.chunks_in_.size() == 0) {
        softmax_.chunks_in_ = {chunks_in_[0]};
        softmax_.chunks_out_ = {prob_};
    }
    softmax_.forward();
    chunks_out_[0]->reshape(1, 1, 1, 1);
    chunks_out_[1]->reshape(logits->shape());
    chunks_out_[1]->copy_from(*prob_);

    int num = logits->num();
    int height = logits->height();
    int width = logits->width();
    float loss = 0;

    const float* labels_data = labels->const_data();
    const float* prob_data = prob_->const_data();
    float* loss_data = chunks_out_[0]->data();
    float* loss_diff = chunks_out_[0]->diff();

    for (int n = 0; n < num; ++n) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                const int lindex = labels->offset(n, 0, h, w);
                const int label_value = static_cast<int>(labels_data[lindex]);
                const int pindex = prob_->offset(n, label_value, h, w);
                loss -= log(max(prob_data[pindex], FLT_MIN));
            }
        }
    }
    loss_data[0] = loss / labels->count();
    loss_diff[0] = 1;
    gradient_reset();
    //cout << "softmaxloss forward" << endl;
}

void SoftmaxLoss::backward() {
    Chunk* logits = chunks_in_[0].get();
    Chunk* labels = chunks_in_[1].get();

    int num = logits->num();
    int height = logits->height();
    int width = logits->width();

    const float* labels_data = labels->const_data();
    const float* prob_data = prob_->const_data();
    const float* loss_diff = chunks_out_[0]->const_diff();
    float* logits_diff = logits->diff();
    memcpy(logits_diff, prob_data, logits->count()*sizeof(float));
    //for (int i = 0; i < logits->count(); ++i) {
    //    logits_diff[i] = prob_data[i];
    //}
    #pragma omp parallel for
    for (int n = 0; n < num; ++n) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                const int laindex = labels->offset(n, 0, h, w);
                const int label_value = static_cast<int>(labels_data[laindex]);
                const int loindex = logits->offset(n, label_value, h, w);
                logits_diff[loindex] -= 1;
            }
        }
    }
    for (int i = 0; i < logits->count(); ++i) {
        logits_diff[i] *= (loss_diff[0] / labels->count());
    }
}

vector<int> SoftmaxLoss::shape_inference() {

}

} // namespace micronet
