#ifndef FOCALLOSS_H
#define FOCALLOSS_H

#include "layer.h"
#include "softmax.h"
#include "chunk.h"

class FocalLoss: public Layer
{
public:
    FocalLoss(const string& layer_name, float gamma = 2);
    virtual void set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks);
    virtual void forward(const vector<Chunk*>& input, const vector<Chunk*>& output);
    virtual void backward(const vector<Chunk*>& input, const vector<Chunk*>& output);
private:
    Softmax softmax_;
    Chunk prob_;
    float gamma_;
};

#endif // FOCALLOSS_H
