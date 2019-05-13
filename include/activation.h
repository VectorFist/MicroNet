#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <set>
#include "layer.h"

namespace micronet {

class Activation: public Layer {
public:
    Activation() = default;
    Activation(const string& activation, float leaky_alpha=0.2, const string& layer_name="activation");
    virtual void forward(bool is_train=true) override;
    virtual void backward() override;
    chunk_ptr operator()(const chunk_ptr& in_chunk);

protected:
    virtual vector<int> shape_inference() override;

private:
    set<string> all_activations_ {"relu", "leaky_relu", "relu6", "prelu", "sigmoid", "tanh",
                                  "elu", "selu", "sin"};
};
} // namespace micronet

#endif // ACTIVATION_H
