#ifndef PADDINGIMAGE_H
#define PADDINGIMAGE_H

#include "layer.h"

namespace micronet {

class PaddingImage: public Layer {
public:
    PaddingImage() {};
    PaddingImage(int padding_top, int padding_bottom, int padding_left,
                 int padding_right, float padding_value = 0.0f,
                 const string& layer_name = "padding_image");
    virtual void forward(bool is_train=true) override;
    virtual void backward() override;
    chunk_ptr operator()(const chunk_ptr& in_chunk);

protected:
    virtual vector<int> shape_inference() override;
};

} //namespace micronet
#endif // PADDINGIMAGE_H
