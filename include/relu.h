#ifndef RELU_H
#define RELU_H
#include "layer.h"

class ReLU: public Layer {
public:
    ReLU(const string& layer_name);
    virtual void set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks);
    virtual void forward(const vector<Chunk*>& input, const vector<Chunk*>& output);
    virtual void backward(const vector<Chunk*>& input, const vector<Chunk*>& output);
};

#endif // RELU_H
