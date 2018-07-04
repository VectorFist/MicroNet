#ifndef BIAS_H
#define BIAS_H
#include "layer.h"

class Bias: public Layer
{
public:
    Bias(int input_channels, const string& layer_name, float bias_value = 0);
    virtual void set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks);
    virtual void forward(const vector<Chunk*>& input, const vector<Chunk*>& output);
    virtual void backward(const vector<Chunk*>& input, const vector<Chunk*>& output);
private:
    void initialize(float bias_value);
};

#endif // BIAS_H
