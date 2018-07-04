#ifndef SOFTMAXLOSS_H
#define SOFTMAXLOSS_H
#
#include "layer.h"
#include "softmax.h"
#include "chunk.h"


class SoftmaxLoss: public Layer
{
public:
    SoftmaxLoss(const string& layer_name);
    virtual void set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks);
    virtual void forward(const vector<Chunk*>& input, const vector<Chunk*>& output);
    virtual void backward(const vector<Chunk*>& input, const vector<Chunk*>& output);
private:
    Softmax softmax_;
    Chunk prob_;
};

#endif // SOFTMAXLOSS_H
