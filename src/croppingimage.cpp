#include <omp.h>
#include "croppingimage.h"

namespace micronet {

CroppingImage::CroppingImage(int cropping_top, int cropping_bottom, int cropping_left,
                             int cropping_right, const string& layer_name):
    Layer(layer_name, "CroppingImage") {
    int_hps_["cropping_top"] = cropping_top;
    int_hps_["cropping_bottom"] = cropping_bottom;
    int_hps_["cropping_left"] = cropping_left;
    int_hps_["cropping_right"] = cropping_right;
}

chunk_ptr CroppingImage::operator()(const chunk_ptr& in_chunk) {
    chunks_in_ = {in_chunk};
    vector<int> out_shape = shape_inference();
    chunk_ptr out_chunk = make_shared<Chunk>(out_shape);
    chunks_out_ = {out_chunk};

    layer_ptr layer = make_shared<CroppingImage>(*this);
    in_chunk->in_layers_.push_back(layer);
    out_chunk->out_layer_ = layer;

    return out_chunk;
}

void CroppingImage::forward(bool is_train) {
    int num = chunks_in_[0]->num();
    int channels = chunks_in_[0]->channels();

    int cropping_top = int_hps_["cropping_top"];
    int cropping_left = int_hps_["cropping_left"];

    vector<int> output_shape = shape_inference();
    chunks_out_[0]->reshape(output_shape);

    const float* input_data = chunks_in_[0]->const_data();
    float* output_data = chunks_out_[0]->data();

    chunk_ptr in_chunk = chunks_in_[0];
    chunk_ptr out_chunk = chunks_out_[0];
    #pragma omp parallel for
    for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int row = 0; row < output_shape[2]; ++row) {
                for (int col = 0; col < output_shape[3]; ++col) {
                    const int iindex = in_chunk->offset(n, c, row+cropping_top, col+cropping_left);
                    const int oindex = out_chunk->offset(n, c, row, col);
                    output_data[oindex] = input_data[iindex];
                }
            }
        }
    }

    gradient_reset();
}

void CroppingImage::backward() {
    int num = chunks_in_[0]->num();
    int channels = chunks_in_[0]->channels();

    int cropping_top = int_hps_["cropping_top"];
    int cropping_left = int_hps_["cropping_left"];

    const float* output_diff = chunks_out_[0]->const_diff();
    float* input_diff = chunks_in_[0]->diff();
    vector<int> output_shape = chunks_out_[0]->shape();

    chunk_ptr in_chunk = chunks_in_[0];
    chunk_ptr out_chunk = chunks_out_[0];
    #pragma omp parallel for
    for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int row = 0; row < output_shape[2]; ++row) {
                for (int col = 0; col < output_shape[3]; ++col) {
                    const int iindex = in_chunk->offset(n, c, row+cropping_top, col+cropping_left);
                    const int oindex = out_chunk->offset(n, c, row, col);
                    input_diff[iindex] += output_diff[oindex];
                }
            }
        }
    }
}

vector<int> CroppingImage::shape_inference() {
    int num = chunks_in_[0]->num();
    int channels = chunks_in_[0]->channels();
    int height = chunks_in_[0]->height();
    int width = chunks_in_[0]->width();

    return {num, channels, height-int_hps_["cropping_top"]-int_hps_["cropping_bottom"],
            width-int_hps_["cropping_left"]-int_hps_["cropping_right"]};
}

} // namespace micronet
