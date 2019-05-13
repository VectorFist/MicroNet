#ifndef L2LOSS_H
#define L2LOSS_H

#include <layer.h>

namespace micronet {

class L2Loss : public Layer {
public:
    L2Loss(const string& layer_name = "l2_loss");
    virtual void forward(bool is_train=true) override;
    virtual void backward() override;
    chunk_ptr operator()(const chunk_ptr& in_chunk1, const chunk_ptr& in_chunk2);

protected:
    virtual vector<int> shape_inference() override;
};

} // namespace micronet

#endif // L2LOSS_H
