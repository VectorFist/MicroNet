#ifndef ACCURACY_H
#define ACCURACY_H
#include "layer.h"
#include "argmax.h"

class Accuracy: public Layer {
public:
    Accuracy(const string& layer_name);
    virtual void set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks);
    virtual void forward(const vector<Chunk*>& input, const vector<Chunk*>& output);
    virtual void backward(const vector<Chunk*>& input, const vector<Chunk*>& output){};
};

#endif // ACCURACY_H
