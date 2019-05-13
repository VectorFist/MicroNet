#ifndef BIAS_H
#define BIAS_H
#include "layer.h"

namespace micronet {

class Bias: public Layer
{
public:
    Bias(int input_channels, const string& layer_name, float bias_value = 0);
    virtual void forward(bool is_train=true) override {};
    virtual void backward() override {};

protected:
    virtual vector<int> shape_inference() override {};
};

} // namespace micronet

#endif // BIAS_H
