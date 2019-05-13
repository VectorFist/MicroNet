/**
 * @file focalloss.cpp
 * @auther yefajie
 * @data 2018/7/23
 **/
#include <cmath>
#include <iostream>
#include "focalloss.h"

namespace micronet {

FocalLoss::FocalLoss(const string& layer_name, float gamma): Layer(layer_name, "FocalLoss"), softmax_("softmax"),
                     prob_(new Chunk()) {
    flt_hps_["gamma"] = gamma;
    cout << "Initialize focalloss layer: " << layer_name << " done..." << endl;
}

vector<chunk_ptr> FocalLoss::operator()(chunk_ptr& in_chunk1, chunk_ptr& in_chunk2) {
    if ((in_chunk1->count() / in_chunk1->channels()) != in_chunk2->count()) {
        cout << "FocalLoss Error: labels shape must match logits shape(" <<
                in_chunk2->str_shape() << "!=" << in_chunk1->str_shape() << ")";
        exit(1);
    }

    chunks_in_ = {in_chunk1, in_chunk2};
    chunk_ptr out_chunk1 = make_shared<Chunk>(1, 1, 1, 1);
    chunk_ptr out_chunk2 = make_shared<Chunk>(in_chunk1->shape());
    chunks_out_ = {out_chunk1, out_chunk2};

    softmax_.chunks_in_ = {in_chunk1};
    softmax_.chunks_out_ = {prob_};

    layer_ptr layer = make_shared<FocalLoss>(*this);
    in_chunk1->in_layers_.push_back(layer);
    in_chunk2->in_layers_.push_back(layer);
    out_chunk1->out_layer_ = layer;
    out_chunk2->out_layer_ = layer;

    return {out_chunk1, out_chunk2};
}

void FocalLoss::forward(bool is_train) {
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

    float gamma = flt_hps_["gamma"];
    for (int n = 0; n < num; ++n) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                const int lindex = labels->offset(n, 0, h, w);
                const int label_value = static_cast<int>(labels_data[lindex]);
                const int pindex = prob_->offset(n, label_value, h, w);
                loss -= pow(1.0 - prob_data[pindex], gamma) * log(max(prob_data[pindex], FLT_MIN));
            }
        }
    }
    loss_data[0] = loss / labels->count();
    loss_diff[0] = 1;
    gradient_reset();
}

void FocalLoss::backward() {
    Chunk* logits = chunks_in_[0].get();
    Chunk* labels = chunks_in_[1].get();

    int num = logits->num();
    int height = logits->height();
    int width = logits->width();

    const float* labels_data = labels->const_data();
    const float* prob_data = prob_->const_data();
    const float* loss_diff = chunks_out_[0]->const_diff();
    float* logits_diff = logits->diff();

    float gamma = flt_hps_["gamma"];
    for (int n = 0; n < num; ++n) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                const int laindex = labels->offset(n, 0, h, w);
                const int label_value = static_cast<int>(labels_data[laindex]);
                float pt = prob_data[prob_->offset(n, label_value, h, w)];
                for (int c = 0; c < logits->channels(); ++c) {
                    const int loindex = logits->offset(n, c, h, w);
                    float pc = prob_data[loindex];
                    if (c == label_value) {
                        logits_diff[loindex] = pow(1 - pt, gamma) *
                            (gamma * pt * log(max(pt, FLT_MIN)) + pt - 1);
                    } else {
                        logits_diff[loindex] = pow(1 - pt, gamma - 1) *
                            (gamma * log(max(pt, FLT_MIN)) * pt * pc) +
                            pow(1 - pt, gamma) * pc;
                    }
                }
            }
        }
    }
    for (int i = 0; i < logits->count(); ++i) {
        logits_diff[i] *= (loss_diff[0] / labels->count());
    }
}

vector<int> FocalLoss::shape_inference() {

}

} // namespace micronet
