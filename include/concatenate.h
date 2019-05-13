#ifndef CONCATENATE_H
#define CONCATENATE_H

#include "layer.h"

namespace micronet {

class Concatenate: public Layer{
public:
    Concatenate(const int axis=1, const string& layer_name="concatenate");
    virtual void forward(bool is_train=true) override;
    virtual void backward() override;
    chunk_ptr operator()(const vector<chunk_ptr>& in_chunks);

protected:
    virtual vector<int> shape_inference() override;
};
} // namspace micronet

#endif // CONCATENATE_H
