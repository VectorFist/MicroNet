#ifndef CHUNK_H
#define CHUNK_H
#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>

#include "nlohmann/json.hpp"

using namespace std;
using json = nlohmann::json;

namespace micronet {

class Layer;
using layer_ptr = shared_ptr<Layer>;

class Chunk {
public:
    Chunk();
    explicit Chunk(const int n, const int c, const int h, const int w);
    explicit Chunk(const vector<int>& shape);
    Chunk(const Chunk& chunk);
    Chunk(Chunk&& chunk);
    ~Chunk();
    Chunk& operator=(const Chunk& chunk);
    Chunk& operator=(Chunk&& chunk);

    void reshape(const int n, const int c, const int h, const int w);
    void reshape(const vector<int>& shape);
    void copy_from(const Chunk& source);
    void fill_value(const float data_value, const float diff_value);

    const float* const_data() const;
    const float* const_diff() const;
    float* data();
    float* diff();
    bool trainable() const {return trainable_;};
    void set_trainable(bool trainable) {trainable_ = trainable;};

    const vector<int> shape() const;
    string str_shape() const;
    string str_shape_exclude(int axis) const;
    inline int shape(int axis) const {return shape_[axis];};
    inline int num() const {return shape_[0];};
    inline int channels() const {return shape_[1];};
    inline int height() const {return shape_[2];};
    inline int width() const {return shape_[3];};
    inline int count() const {return shape_[0] * shape_[1] * shape_[2] * shape_[3];};
    inline int offset(int n, int c, int h, int w) const {return ((n * channels() + c) * height() + h) * width() + w;};

    vector<layer_ptr> in_layers_;
    layer_ptr out_layer_;

    vector<int> shape_;

private:
    void new_chunk(const vector<int>& shape);
    void delete_chunk();

private:
    //shared_ptr<vector<float> > data_;
    //shared_ptr<vector<float> > diff_;
    float* data_;
    float* diff_;
    bool trainable_ = true;

    friend shared_ptr<Chunk> parse_param(const json& j_param, map<string, shared_ptr<Chunk>>& params);
};

} // namespace micronet


#endif // CHUNK_H
