#ifndef RESHAPE_H
#define RESHAPE_H

#include "layer.h"

namespace micronet {

class Reshape: public Layer {
public:
    Reshape(int n, int c, int h, int w, const string& layer_name = "reshape");
    Reshape() = default;
    virtual void forward(bool is_train=true) override;
    virtual void backward() override;
    chunk_ptr operator()(const chunk_ptr& in_chunk);

protected:
    virtual vector<int> shape_inference() override;

};
} // namespace micronet

#endif // RESHAPE_H
