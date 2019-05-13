#include <cstring>
#include "reshape.h"
#include "math_func.h"

namespace micronet {

Reshape::Reshape(int n, int c, int h, int w, const string& layer_name):
    Layer(layer_name, "Reshape") {
    int_hps_["n"] = n;
    int_hps_["c"] = c;
    int_hps_["h"] = h;
    int_hps_["w"] = w;
}

chunk_ptr Reshape::operator()(const chunk_ptr& in_chunk) {
    chunks_in_ = {in_chunk};
    chunk_ptr out_chunk = make_shared<Chunk>(shape_inference());
    chunks_out_ = {out_chunk};

    layer_ptr layer = make_shared<Reshape>(*this);
    in_chunk->in_layers_.push_back(layer);
    out_chunk->out_layer_ = layer;

    return out_chunk;
}

void Reshape::forward(bool is_train) {
    auto in_chunk = chunks_in_[0];
    auto out_chunk = chunks_out_[0];
    out_chunk->reshape(shape_inference());

    const float* in_data = in_chunk->const_data();
    float* out_data = out_chunk->data();

    memcpy(out_data, in_data, out_chunk->count()*sizeof(float));
    gradient_reset();
}

void Reshape::backward() {
    auto in_chunk = chunks_in_[0];
    auto out_chunk = chunks_out_[0];
    const float* out_diff = out_chunk->const_diff();
    float* in_diff = in_chunk->diff();

    add(out_chunk->count(), out_diff, 1.0f, in_diff, 1.0f, in_diff);
}

vector<int> Reshape::shape_inference() {
    int n = int_hps_["n"];
    int c = int_hps_["c"];
    int h = int_hps_["h"];
    int w = int_hps_["w"];
    int in_count = chunks_in_[0]->count();
    if (n == -1) {
        return {in_count/(c*h*w), c, h, w};
    } else if (c == -1) {
        return {n, in_count/(n*h*w), h, w};
    } else if (h == -1) {
        return {n, c, in_count/(n*c*w), w};
    } else if (w == -1) {
        return {n, c, h, in_count/(n*c*h)};
    } else if (n*c*h*w == in_count) {
        return {n, c, h, w};
    } else {
        cout << "Reshape error" << endl;
        exit(1);
    }
}

} // namespace micronet
