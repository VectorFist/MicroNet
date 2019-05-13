#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "layer.h"

namespace micronet {

class Softmax: public Layer {
public:
    Softmax(const string& layer_name="softmax");
    virtual void forward(bool is_train=true) override;
    virtual void backward() override {};
    chunk_ptr operator()(chunk_ptr& in_chunk);

protected:
    virtual vector<int> shape_inference() override;

    friend class FocalLoss;
    friend class SoftmaxLoss;
};
} // namespace micronet

#endif // SOFTMAX_H
