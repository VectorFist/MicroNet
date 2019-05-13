#ifndef ADD_H
#define ADD_H

#include <string>
#include "layer.h"

namespace micronet {

class Add : public Layer
{
public:
    Add(const string& layer_name="add");
    virtual void forward(bool is_train=true) override;
    virtual void backward() override;
    chunk_ptr operator()(const chunk_ptr& in_chunk1, const chunk_ptr& in_chunk2);

protected:
    virtual vector<int> shape_inference() override;
};
} // namespace micronet

#endif // ADD_H
