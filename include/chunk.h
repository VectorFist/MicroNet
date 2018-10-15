#ifndef CHUNK_H
#define CHUNK_H
#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <sstream>

using namespace std;


class Chunk {
public:
    Chunk();
    explicit Chunk(const int n, const int c, const int h, const int w);
    explicit Chunk(const vector<int>& shape);
    Chunk(const Chunk& chunk);
    Chunk& operator=(const Chunk& chunk);

    const float* const_data() const;
    const float* const_diff() const;
    float* data();
    float* diff();

    void reshape(const int n, const int c, const int h, const int w);
    void reshape(const vector<int>& shape);
    void copy_from(const Chunk& source);

    const vector<int>& shape() const;
    string str_shape() const;
    inline int shape(int axis) const {return shape_[axis];};
    inline int num() const {return shape_[0];};
    inline int channels() const {return shape_[1];};
    inline int height() const {return shape_[2];};
    inline int width() const {return shape_[3];};
    inline int count() const {return shape_[0] * shape_[1] * shape_[2] * shape_[3];};
    inline int offset(int n, int c, int h, int w) const {return ((n * channels() + c) * height() + h) * width() + w;};

private:
    shared_ptr<vector<float> > data_;
    shared_ptr<vector<float> > diff_;
    vector<int> shape_;
};


#endif // CHUNK_H
