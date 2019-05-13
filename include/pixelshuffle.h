#ifndef PIXELSHUFFLE_H
#define PIXELSHUFFLE_H

#include "layer.h"

namespace micronet {

class PixelShuffle: public Layer {
public:
    PixelShuffle(int upscale_factor, const string& layer_name = "pixel_shuffle");
    PixelShuffle() = default;
    virtual void forward(bool is_train=true) override;
    virtual void backward() override;
    chunk_ptr operator()(const chunk_ptr& in_chunk);

protected:
    virtual vector<int> shape_inference() override;

};
} // namespace micronet

#endif // PIXELSHUFFLE_H
