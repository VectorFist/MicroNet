#include <omp.h>
#include "paddingimage.h"

namespace micronet {

PaddingImage::PaddingImage(int padding_top, int padding_bottom, int padding_left,
                           int padding_right, float padding_value, const string& layer_name):
    Layer(layer_name, "PaddingImage") {
    int_hps_["padding_top"] = padding_top;
    int_hps_["padding_bottom"] = padding_bottom;
    int_hps_["padding_left"] = padding_left;
    int_hps_["padding_right"] = padding_right;
    flt_hps_["padding_value"] = padding_value;
}

chunk_ptr PaddingImage::operator()(const chunk_ptr& in_chunk) {
    chunks_in_ = {in_chunk};
    vector<int> out_shape = shape_inference();
    chunk_ptr out_chunk = make_shared<Chunk>(out_shape);
    chunks_out_ = {out_chunk};

    layer_ptr layer = make_shared<PaddingImage>(*this);
    in_chunk->in_layers_.push_back(layer);
    out_chunk->out_layer_ = layer;

    return out_chunk;
}

void PaddingImage::forward(bool is_train) {
    int num = chunks_in_[0]->num();
    int channels = chunks_in_[0]->channels();
    int height = chunks_in_[0]->height();
    int width = chunks_in_[0]->width();

    int padding_top = int_hps_["padding_top"];
    int padding_left = int_hps_["padding_left"];
    float padding_value = flt_hps_["padding_value"];

    vector<int> output_shape = shape_inference();
    chunks_out_[0]->reshape(output_shape);

    const float* input_data = chunks_in_[0]->const_data();
    float* output_data = chunks_out_[0]->data();
    chunks_out_[0]->fill_value(padding_value, 0.0f);

    chunk_ptr in_chunk = chunks_in_[0];
    chunk_ptr out_chunk = chunks_out_[0];
    #pragma omp parallel for
    for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int row = 0; row < height; ++row) {
                for (int col = 0; col < width; ++col) {
                    const int iindex = in_chunk->offset(n, c, row, col);
                    const int oindex = out_chunk->offset(n, c, row+padding_top, col+padding_left);
                    output_data[oindex] = input_data[iindex];
                }
            }
        }
    }

    gradient_reset();
}

void PaddingImage::backward() {
    int num = chunks_in_[0]->num();
    int channels = chunks_in_[0]->channels();
    int height = chunks_in_[0]->height();
    int width = chunks_in_[0]->width();

    int padding_top = int_hps_["padding_top"];
    int padding_left = int_hps_["padding_left"];

    const float* output_diff = chunks_out_[0]->const_diff();
    float* input_diff = chunks_in_[0]->diff();

    chunk_ptr in_chunk = chunks_in_[0];
    chunk_ptr out_chunk = chunks_out_[0];
    #pragma omp parallel for
    for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int row = 0; row < height; ++row) {
                for (int col = 0; col < width; ++col) {
                    const int iindex = in_chunk->offset(n, c, row, col);
                    const int oindex = out_chunk->offset(n, c, row+padding_top, col+padding_left);
                    input_diff[iindex] += output_diff[oindex];
                }
            }
        }
    }
}

vector<int> PaddingImage::shape_inference() {
    int num = chunks_in_[0]->num();
    int channels = chunks_in_[0]->channels();
    int height = chunks_in_[0]->height();
    int width = chunks_in_[0]->width();

    return {num, channels, height+int_hps_["padding_top"]+int_hps_["padding_bottom"],
            width+int_hps_["padding_left"]+int_hps_["padding_right"]};
}

} // namespace micronet
