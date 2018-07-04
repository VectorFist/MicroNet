#ifndef ARGMAX_H
#define ARGMAX_H
#include "layer.h"

class ArgMax: public Layer {
public:
    ArgMax(const string& layer_name);
    virtual void set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks);
    virtual void forward(const vector<Chunk*>& input, const vector<Chunk*>& output);
    virtual void backward(const vector<Chunk*>& input, const vector<Chunk*>& output){};
};

#endif // ARGMAX_H
