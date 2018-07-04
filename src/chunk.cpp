/**
 * @file chunk.cpp
 * @auther yefajie
 * @data 2018/6/21
 **/
#include "chunk.h"

Chunk::Chunk() {
    shape_ = {0, 0, 0, 0};
    data_ = make_shared<vector<float> >();
    diff_ = make_shared<vector<float> >();
}

Chunk::Chunk(const int n, const int c, const int h, const int w) {
    shape_ = {n, c, h, w};
    data_ = make_shared<vector<float> >(count());
    diff_ = make_shared<vector<float> >(count());
}

Chunk::Chunk(const vector<int>& shape) {
    shape_ = shape;
    data_ = make_shared<vector<float> >(count());
    diff_ = make_shared<vector<float> >(count());
}

const float* Chunk::const_data() const {
    return data_->data();
}

const float* Chunk::const_diff() const {
    return diff_->data();
}

float* Chunk::data() {
    return data_->data();
}

float* Chunk::diff() {
    return diff_->data();
}

void Chunk::reshape(const int n, const int c, const int h, const int w) {
    shape_ = {n, c, h, w};
    data_->resize(count());
    diff_->resize(count());
}

void Chunk::reshape(const vector<int>& shape) {
    shape_ = shape;
    data_->resize(count());
    diff_->resize(count());
}

void Chunk::copy_from(const Chunk& source) {
    shape_ = source.shape();
    data_->assign(source.const_data(), source.const_data() + source.count());
    diff_->assign(source.const_diff(), source.const_diff() + source.count());
}

const vector<int>& Chunk::shape() const {
    return shape_;
}

string Chunk::str_shape() const {
    stringstream str;
    str << "(" << shape_[0] << ", " << shape_[1] << ", " << shape_[2] << ", " << shape_[3] << ")";
    return str.str();
}
