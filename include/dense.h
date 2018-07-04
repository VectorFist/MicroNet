#ifndef DENSE_H
#define DENSE_H
#include "layer.h"

class Dense: public Layer
{
public:
    Dense(int in_dim, int out_dim, const string& layer_name, float mean = 0, float stddev = 0.1, float bias_value = 0.1);
    virtual void set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks);
    virtual void forward(const vector<Chunk*>& input, const vector<Chunk*>& output);
    virtual void backward(const vector<Chunk*>& input, const vector<Chunk*>& output);
private:
    void initialize(float mean, float stddev, float bias_value);
};

#endif // DENSE_H
