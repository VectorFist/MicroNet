#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include "chunk.h"

#define FLT_MAX 3.402823466e+30F
#define FLT_MIN 1.175494351e-30F

using namespace std;

class Layer {
public:
    Layer(){};
    virtual ~Layer(){};
    virtual void set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks) = 0;
    virtual void forward(const vector<Chunk*>& input, const vector<Chunk*>& output) = 0;
    virtual void backward(const vector<Chunk*>& input, const vector<Chunk*>& output) = 0;
    vector<shared_ptr<Chunk>> params_;
    vector<string> in_chunks_, out_chunks_;
    string layer_name_;
};

#endif // LAYER_H
