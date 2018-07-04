#ifndef POOLING_H
#define POOLING_H
#include <string>
#include "layer.h"

class Pooling: public Layer {
public:
    Pooling(int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h,
            int stride_w, const string& layer_name, const string& pooling = "max");
    virtual void set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks);
    virtual void forward(const vector<Chunk*>& input, const vector<Chunk*>& output);
    virtual void backward(const vector<Chunk*>& input, const vector<Chunk*>& output);
private:
    int kernel_h_, kernel_w_;
    int pad_h_, pad_w_;
    int stride_h_, stride_w_;
    string pooling_;
    Chunk mask_;
};

#endif // POOLING_H
