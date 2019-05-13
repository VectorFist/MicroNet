#include "pixelshuffle.h"

namespace micronet {

PixelShuffle::PixelShuffle(int upscale_factor, const string& layer_name):
    Layer(layer_name, "PixelShuffle") {
    int_hps_["upscale_factor"] = upscale_factor;
}

chunk_ptr PixelShuffle::operator()(const chunk_ptr& in_chunk) {
    int upscale_factor = int_hps_["upscale_factor"];
    if (in_chunk->channels() % (upscale_factor * upscale_factor) != 0) {
        cout << "inchunk channels error" <<  endl;
        exit(1);
    }

    chunks_in_ = {in_chunk};
    chunk_ptr out_chunk = make_shared<Chunk>(shape_inference());
    chunks_out_ = {out_chunk};

    layer_ptr layer = make_shared<PixelShuffle>(*this);
    in_chunk->in_layers_.push_back(layer);
    out_chunk->out_layer_ = layer;

    return out_chunk;
}

void PixelShuffle::forward(bool is_train) {
    auto in_chunk = chunks_in_[0];
    auto out_chunk = chunks_out_[0];
    out_chunk->reshape(shape_inference());

    const float* in_data = in_chunk->const_data();
    float* out_data = out_chunk->data();
    int upscale_factor = int_hps_["upscale_factor"];

    for (int n = 0; n < in_chunk->num(); ++n) {
        for (int c = 0; c < in_chunk->channels(); ++c) {
            for (int h = 0; h < in_chunk->height(); ++h) {
                for (int w = 0; w < in_chunk->width(); ++w) {
                    const int oc = c / (upscale_factor * upscale_factor);
                    const int ic_tmp = c % (upscale_factor * upscale_factor);
                    const int oh = h * upscale_factor + ic_tmp / upscale_factor;
                    const int ow = w * upscale_factor + ic_tmp % upscale_factor;

                    const int iindex = in_chunk->offset(n, c, h, w);
                    const int oindex = out_chunk->offset(n, oc, oh, ow);
                    out_data[oindex] = in_data[iindex];
                }
            }
        }
    }
    gradient_reset();
}

void PixelShuffle::backward() {
    auto in_chunk = chunks_in_[0];
    auto out_chunk = chunks_out_[0];

    const float* out_diff = out_chunk->const_diff();
    float* in_diff = in_chunk->diff();
    int upscale_factor = int_hps_["upscale_factor"];

    for (int n = 0; n < in_chunk->num(); ++n) {
        for (int c = 0; c < in_chunk->channels(); ++c) {
            for (int h = 0; h < in_chunk->height(); ++h) {
                for (int w = 0; w < in_chunk->width(); ++w) {
                    const int oc = c / (upscale_factor * upscale_factor);
                    const int ic_tmp = c % (upscale_factor * upscale_factor);
                    const int oh = h * upscale_factor + ic_tmp / upscale_factor;
                    const int ow = w * upscale_factor + ic_tmp % upscale_factor;

                    const int iindex = in_chunk->offset(n, c, h, w);
                    const int oindex = out_chunk->offset(n, oc, oh, ow);
                    in_diff[iindex] += out_diff[oindex];
                }
            }
        }
    }
}

vector<int> PixelShuffle::shape_inference() {
    int n = chunks_in_[0]->num();
    int c = chunks_in_[0]->channels();
    int h = chunks_in_[0]->height();
    int w = chunks_in_[0]->width();
    int upscale_factor = int_hps_["upscale_factor"];

    int oc = c / upscale_factor / upscale_factor;
    int oh = h * upscale_factor;
    int ow = w * upscale_factor;

    return {n, oc, oh, ow};
}

} // namespace micronet
