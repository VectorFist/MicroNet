#ifndef ARGMAX_H
#define ARGMAX_H
#include "layer.h"

namespace micronet {

class ArgMax: public Layer {
public:
    ArgMax(const string& layer_name = "argmax");
    virtual void forward(bool is_train=true) override;
    virtual void backward() override {};
    chunk_ptr operator()(const chunk_ptr& in_chunk);

protected:
    virtual vector<int> shape_inference() override;

    friend class Accuracy;
};
} // namespace micronet

#endif // ARGMAX_H
