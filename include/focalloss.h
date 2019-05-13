#ifndef FOCALLOSS_H
#define FOCALLOSS_H

#include "layer.h"
#include "softmax.h"
#include "chunk.h"

namespace micronet {

class FocalLoss: public Layer
{
public:
    FocalLoss(const string& layer_name = "focal_loss", float gamma = 2);
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

#endif // FOCALLOSS_H
