#ifndef CROPPINGIMAGE_H
#define CROPPINGIMAGE_H

#include "layer.h"

namespace micronet {

class CroppingImage: public Layer {
public:
    CroppingImage() {};
    CroppingImage(int cropping_top, int cropping_bottom, int cropping_left,
                  int cropping_right, const string& layer_name = "cropping_image");
    virtual void forward(bool is_train=true) override;
    virtual void backward() override;
    chunk_ptr operator()(const chunk_ptr& in_chunk);

protected:
    virtual vector<int> shape_inference() override;
};

} //namespace micronet

#endif // CROPPINGIMAGE_H
