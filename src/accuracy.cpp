/**
 * @file accuracy.cpp
 * @auther yefajie
 * @data 2018/6/24
 **/
#include "accuracy.h"
#include <iostream>

Accuracy::Accuracy(const string& layer_name) {
    layer_name_ = layer_name;
    cout << "Initialize acc layer: " << layer_name_ << " done..." << endl;
}

void Accuracy::set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks) {
    in_chunks_ = in_chunks;
    out_chunks_ = out_chunks;
}

void Accuracy::forward(const vector<Chunk*>& input, const vector<Chunk*>& output) {
    Chunk* prob = input[0];
    Chunk* labels = input[1];
    if ((prob->count() / prob->channels()) != labels->count()) {
        cout << "Accuracy Error: labels shape must match prob shape(" << labels->str_shape() << "!=" << prob->str_shape() << ")";
        exit(1);
    }

    Chunk predictions;
    ArgMax argmax("argmax");
    argmax.forward(vector<Chunk*>{prob}, vector<Chunk*>{&predictions});

    output[0]->reshape(1, 1, 1, 1);
    const float* labels_data = labels->const_data();
    const float* predictions_data = predictions.const_data();
    float* output_data = output[0]->data();

    int corr_count = 0;
    for (int i = 0; i < labels->count(); ++i) {
        if (labels_data[i] == predictions_data[i]) {
            corr_count++;
        }
    }
    output_data[0] = float(corr_count) / labels->count();
}
