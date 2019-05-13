/**
 * @file chunk.cpp
 * @auther yefajie
 * @data 2018/6/21
 **/
#include "chunk.h"

namespace micronet {

void Chunk::new_chunk(const vector<int>& shape) {
    delete_chunk();
    shape_ = shape;
    data_ = new float[count()];
    diff_ = new float[count()];
    //cout << "new chunk" << endl;
}

void Chunk::delete_chunk() {
    if (data_ != nullptr) {
        delete[] data_;
        data_ = nullptr;
    }
    if (diff_ != nullptr) {
        delete[] diff_;
        diff_ = nullptr;
    }
    shape_ = {0, 0, 0, 0};
}

Chunk::Chunk(): shape_{0, 0, 0, 0}, data_(nullptr), diff_(nullptr) {
}

Chunk::Chunk(const int n, const int c, const int h, const int w):
    shape_{n, c, h, w}, data_(new float[count()]), diff_(new float[count()]) {
    fill_value(0.0f, 0.0f);
}

Chunk::Chunk(const vector<int>& shape): shape_(shape),
    data_(new float[count()]), diff_(new float[count()]) {
    fill_value(0.0f, 0.0f);
}

Chunk::Chunk(const Chunk& chunk): shape_(chunk.shape()), data_(new float[count()]), diff_(new float[count()]),
    in_layers_(chunk.in_layers_), out_layer_(chunk.out_layer_), trainable_(chunk.trainable()) {
    std::copy(chunk.const_data(), chunk.const_data()+chunk.count(), data_);
    std::copy(chunk.const_diff(), chunk.const_diff()+chunk.count(), diff_);
}

Chunk::Chunk(Chunk&& chunk): shape_(chunk.shape()), data_(chunk.data()), diff_(chunk.diff()),
    in_layers_(chunk.in_layers_), out_layer_(chunk.out_layer_), trainable_(chunk.trainable()) {
    chunk.shape_ = {0, 0, 0, 0};
    chunk.data_ = nullptr;
    chunk.diff_ = nullptr;
}

Chunk::~Chunk() {
    delete_chunk();
}

Chunk& Chunk::operator=(const Chunk& chunk) {
    new_chunk(chunk.shape());
    std::copy(chunk.const_data(), chunk.const_data()+chunk.count(), data_);
    std::copy(chunk.const_diff(), chunk.const_diff()+chunk.count(), diff_);
    in_layers_ = chunk.in_layers_;
    out_layer_ = chunk.out_layer_;
    trainable_ = chunk.trainable();
    return *this;
}

Chunk& Chunk::operator=(Chunk&& chunk) {
    if (&chunk != this) {
        delete_chunk();
        shape_ = chunk.shape();
        in_layers_ = chunk.in_layers_;
        out_layer_ = chunk.out_layer_;
        trainable_ = chunk.trainable();
        data_ = chunk.data();
        diff_ = chunk.diff();

        chunk.shape_ = {0, 0, 0, 0};
        chunk.data_ = nullptr;
        chunk.data_ = nullptr;
    }
    return *this;
}

const float* Chunk::const_data() const {
    return data_;
}

const float* Chunk::const_diff() const {
    return diff_;
}

float* Chunk::data() {
    return data_;
}

float* Chunk::diff() {
    return diff_;
}

void Chunk::reshape(const int n, const int c, const int h, const int w) {
    if (count() == n*c*h*w) {
        shape_ = {n, c, h, w};
        return;
    }
    reshape({n, c, h, w});
}

void Chunk::reshape(const vector<int>& shape) {
    if (count() == shape[0]*shape[1]*shape[2]*shape[3]) {
        shape_ = shape;
        return;
    }
    new_chunk(shape);
    fill_value(0.0f, 0.0f);
}

void Chunk::copy_from(const Chunk& source) {
    new_chunk(source.shape());
    std::copy(source.const_data(), source.const_data()+source.count(), data_);
    std::copy(source.const_diff(), source.const_diff()+source.count(), diff_);
}

void Chunk::fill_value(const float data_value, const float diff_value) {
    std::fill(data_, data_+count(), data_value);
    std::fill(diff_, diff_+count(), diff_value);
}

const vector<int> Chunk::shape() const {
    return shape_;
}

string Chunk::str_shape() const {
    stringstream str;
    str << "(" << shape_[0] << ", " << shape_[1] << ", " << shape_[2] << ", " << shape_[3] << ")";
    return str.str();
}

string Chunk::str_shape_exclude(int axis) const {
    stringstream str;
    switch(axis) {
        case 0: str << shape_[1] << ", " << shape_[2] << ", " << shape_[3]; break;
        case 1: str << shape_[0] << ", " << shape_[2] << ", " << shape_[3]; break;
        case 2: str << shape_[0] << ", " << shape_[1] << ", " << shape_[3]; break;
        case 3: str << shape_[0] << ", " << shape_[1] << ", " << shape_[2]; break;
    }
    return str.str();
}

} // namespace micronet
