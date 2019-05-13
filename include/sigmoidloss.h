#ifndef SIGMOIDLOSS_H
#define SIGMOIDLOSS_H

#include "layer.h"
#include "chunk.h"

namespace micronet {

class SigmoidLoss: public Layer
{
public:
    SigmoidLoss(const string& layer_name="sigmoid_loss");
    virtual void forward(bool is_train=true) override;
    virtual void backward() override;
    vector<chunk_ptr> operator()(chunk_ptr& in_chunk1, chunk_ptr& in_chunk2);

protected:
    virtual vector<int> shape_inference() override;

};
} // namespace micronet


#endif // SIGMOIDLOSS_H
