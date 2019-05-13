/**
 * @file dataprovider.cpp
 * @auther yefajie
 * @data 2018/6/22
 **/
#include <string.h>

#include "dataprovider.h"
#include "util.h"

namespace micronet {

DataProvider::DataProvider(vector<data_t>& data_vec, bool shuffle):
    data_vec_(std::move(data_vec)), shuffle_(shuffle), index_in_epoch_(0) {
    num_samples_ = data_vec_[0].size();
    if (shuffle_) {
        shuffle_data();
    }
}

void DataProvider::load_batch(const vector<chunk_ptr>& chunks_in, int batch_size) {
    if (index_in_epoch_ + batch_size > num_samples_) {
        if (shuffle_) {
            shuffle_data();
        }
        index_in_epoch_ = 0;
    }
    int load_start = index_in_epoch_;
    int load_end = load_start + batch_size;
    index_in_epoch_ += batch_size;

    for (int i = 0; i < chunks_in.size(); ++i) {
        const chunk_ptr& chunk = chunks_in[i];
        data_t& data = data_vec_[i];
        chunk->reshape(batch_size, chunk->channels(), chunk->height(), chunk->width());

        int dim = chunk->channels() * chunk->height() * chunk->width();
        if (dim != data[0].size()) {
            cout << "Input Chunk " << i+1 << " shape must match Input Data dim..." << endl;
            exit(1);
        }

        float* chunk_data = chunk->data();
        for (int n = load_start; n < load_end; ++n) {
            memcpy(chunk_data, data[n].data(), dim*sizeof(float));
            chunk_data += dim;
        }
    }
}

void DataProvider::shuffle_data() {
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    for (auto& data: data_vec_) {
        shuffle(data.begin(), data.end(), default_random_engine(seed));
    }
    cout << "Shuffle data done, load batch data from a new epoch..." << endl;
}

} // namespace micronet
