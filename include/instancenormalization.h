#ifndef INSTANCENORMALIZATION_H
#define INSTANCENORMALIZATION_H

#include "layer.h"

namespace micronet {

class InstanceNormalization: public Layer {
public:
    InstanceNormalization(const string& layer_name = "instance_normalization");
    virtual void forward(bool is_train=true) override;
    virtual void backward() override;
    chunk_ptr operator()(const chunk_ptr& in_chunk);

protected:
    virtual vector<int> shape_inference() override;

private:
    void initialize();
    chunk_ptr mean_, var_, out_no_shift_;

};
} // namespace micronet

#endif // INSTANCENORMALIZATION_H
