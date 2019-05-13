#ifndef ACCURACY_H
#define ACCURACY_H
#include "layer.h"
#include "argmax.h"

namespace micronet {

class Accuracy: public Layer {
public:
    Accuracy(const string& layer_name="accuracy");
    virtual void forward(bool is_train=true) override;
    virtual void backward()override {};
    chunk_ptr operator()(const chunk_ptr& in_chunk1, const chunk_ptr& in_chunk2);

protected:
    virtual vector<int> shape_inference() override;
};
} // namespace micronet

#endif // ACCURACY_H
