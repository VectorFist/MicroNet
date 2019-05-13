#ifndef DENSE_H
#define DENSE_H
#include "layer.h"

namespace micronet {

class Dense: public Layer
{
public:
    Dense() = default;
    Dense(int out_dim, float mean = 0, float stddev = 0.1, float bias_value = 0.1, const string& layer_name="dense");
    virtual void forward(bool is_train=true) override;
    virtual void backward() override;
    chunk_ptr operator()(const chunk_ptr& in_chunk);

protected:
    virtual vector<int> shape_inference() override;

private:
    void initialize();
    Chunk all_one_tmp_;
};
} // namespace micornet

#endif // DENSE_H
