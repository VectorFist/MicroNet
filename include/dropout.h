#ifndef DROPOUT_H
#define DROPOUT_H

#include "layer.h"

namespace micronet {

class Dropout: public Layer {
public:
    Dropout(float keep_prob = 0.5, const string& layer_name = "dropout");
    virtual void forward(bool is_train=true) override;
    virtual void backward() override;
    chunk_ptr operator()(const chunk_ptr& in_chunk);

protected:
    virtual vector<int> shape_inference() override;

private:
    chunk_ptr mask_;

};
} // namespace micronet

#endif // DROPOUT_H
