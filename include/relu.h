#ifndef RELU_H
#define RELU_H
#include "layer.h"

namespace micronet {

class ReLU: public Layer {
public:
    ReLU(const string& layer_name = "relu");
    virtual void forward(bool is_train=true) override {};
    virtual void backward() override {};
    chunk_ptr operator()(chunk_ptr& in_chunk);

protected:
    virtual vector<int> shape_inference() override {};

};
} // namespace micronet

#endif // RELU_H
