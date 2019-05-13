#ifndef BATCHMIDDLESPLIT_H
#define BATCHMIDDLESPLIT_H

#include <utility>
#include "layer.h"

namespace micronet {

class BatchMiddleSplit : public Layer {
public:
    BatchMiddleSplit(const string& layer_name="batch_middle_split");
    virtual void forward(bool is_train=true) override;
    virtual void backward() override;
    pair<chunk_ptr, chunk_ptr> operator()(const chunk_ptr& in_chunk);

protected:
    virtual vector<int> shape_inference() override;
};

} // namespace micronet

#endif // BATCHMIDDLESPLIT_H
