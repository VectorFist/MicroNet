#ifndef DATAPROVIDER_H
#define DATAPROVIDER_H
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <memory>

#include "chunk.h"

using namespace std;

namespace micronet {

using data_t = vector<vector<float>>;
using chunk_ptr = shared_ptr<Chunk>;


class DataProvider {
public:
    DataProvider(vector<data_t>& data_vec, bool shuffle = true);
    void load_batch(const vector<chunk_ptr>& chunks_in, int batch_size);
    inline int num_samples() {return num_samples_;};
private:
    void shuffle_data();
    vector<data_t> data_vec_;
    bool shuffle_;
    int index_in_epoch_;
    int num_samples_;
};
} //namespace micronet

#endif // DATAPROVIDER_H
