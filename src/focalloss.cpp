/**
 * @file focalloss.cpp
 * @auther yefajie
 * @data 2018/7/23
 **/
#include <cmath>
#include <iostream>
#include "focalloss.h"

FocalLoss::FocalLoss(const string& layer_name, float gamma): softmax_("softmax") {
    layer_name_ = layer_name;
    gamma_ = gamma;
    cout << "Initialize focalloss layer: " << layer_name_ << " done..." << endl;
}

void FocalLoss::set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks) {
    in_chunks_ = in_chunks;
    out_chunks_ = out_chunks;
}

void FocalLoss::forward(const vector<Chunk*>& input, const vector<Chunk*>& output) {
    Chunk* logits = input[0];
    Chunk* labels = input[1];
    if ((logits->count() / logits->channels()) != labels->count()) {
        cout << "FocalLoss Error: labels shape must match logits shape(" << labels->str_shape() << "!=" << logits->str_shape() << ")";
        exit(1);
    }
    softmax_.forward(vector<Chunk*>{logits}, vector<Chunk*>{&prob_});
    output[0]->reshape(1, 1, 1, 1);
    output[1]->reshape(logits->shape());
    output[1]->copy_from(prob_);

    int num = logits->num();
    int height = logits->height();
    int width = logits->width();
    float loss = 0;

    const float* labels_data = labels->const_data();
    const float* prob_data = prob_.const_data();
    float* loss_data = output[0]->data();
    float* loss_diff = output[0]->diff();

    for (int n = 0; n < num; ++n) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                const int lindex = labels->offset(n, 0, h, w);
                const int label_value = static_cast<int>(labels_data[lindex]);
                const int pindex = prob_.offset(n, label_value, h, w);
                loss -= pow(1.0 - prob_data[pindex], gamma_) * log(max(prob_data[pindex], FLT_MIN));
            }
        }
    }
    loss_data[0] = loss / labels->count();
    loss_diff[0] = 1;
}

void FocalLoss::backward(const vector<Chunk*>& input, const vector<Chunk*>& output) {
    Chunk* logits = input[0];
    Chunk* labels = input[1];

    int num = logits->num();
    int height = logits->height();
    int width = logits->width();

    const float* labels_data = labels->const_data();
    const float* prob_data = prob_.const_data();
    const float* loss_diff = output[0]->const_diff();
    float* logits_diff = logits->diff();

    for (int n = 0; n < num; ++n) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                const int laindex = labels->offset(n, 0, h, w);
                const int label_value = static_cast<int>(labels_data[laindex]);
                float pt = prob_data[prob_.offset(n, label_value, h, w)];
                for (int c = 0; c < logits->channels(); ++c) {
                    const int loindex = logits->offset(n, c, h, w);
                    float pc = prob_data[loindex];
                    if (c == label_value) {
                        logits_diff[loindex] = pow(1 - pt, gamma_) *
                            (gamma_ * pt * log(max(pt, FLT_MIN)) + pt - 1);
                    } else {
                        logits_diff[loindex] = pow(1 - pt, gamma_ - 1) *
                            (-gamma_ * log(max(pt, FLT_MIN)) * pt * pc) +
                            pow(1 - pt, gamma_) * pc;
                    }
                }
            }
        }
    }
    for (int i = 0; i < logits->count(); ++i) {
        logits_diff[i] *= (loss_diff[0] / labels->count());
    }
}
