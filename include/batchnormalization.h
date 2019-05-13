#ifndef BATCHNORMALIZATION_H
#define BATCHNORMALIZATION_H


#include "layer.h"

namespace micronet {

class BatchNormalization: public Layer {
public:
    BatchNormalization(const string& layer_name = "batch_normalization");
    virtual void forward(bool is_train=true) override;
    virtual void backward() override;
    chunk_ptr operator()(const chunk_ptr& in_chunk);

protected:
    virtual vector<int> shape_inference() override;

private:
    void initialize();
    chunk_ptr out_no_shift_;

};
} // namespace micronet

#endif // BATCHNORMALIZATION_H
