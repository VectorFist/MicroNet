#ifndef SOFTMAXLOSS_H
#define SOFTMAXLOSS_H

#include "layer.h"
#include "softmax.h"
#include "chunk.h"

namespace micronet {

class SoftmaxLoss: public Layer
{
public:
    SoftmaxLoss(const string& layer_name="softmax_loss");
    virtual void forward(bool is_train=true) override;
    virtual void backward() override;
    vector<chunk_ptr> operator()(chunk_ptr& in_chunk1, chunk_ptr& in_chunk2);

protected:
    virtual vector<int> shape_inference() override;

private:
    Softmax softmax_;
    chunk_ptr prob_;
};
} // namespace micronet

#endif // SOFTMAXLOSS_H
