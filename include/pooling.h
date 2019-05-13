#ifndef POOLING_H
#define POOLING_H
#include <string>
#include "layer.h"

namespace micronet {

class Pooling: public Layer {
public:
    Pooling(): mask_(new Chunk) {};
    Pooling(int kernel_h, int kernel_w, int stride_h, int stride_w, const string& padding = "valid",
            const string& pooling = "max", const string& layer_name = "pooling");
    virtual void forward(bool is_train=true) override;
    virtual void backward() override;
    chunk_ptr operator()(chunk_ptr& in_chunk);

protected:
    virtual vector<int> shape_inference() override;

private:
    void pad_inference();
    chunk_ptr mask_;
};
} // namespace micronet

#endif // POOLING_H
