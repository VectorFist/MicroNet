#ifndef DATAPROVIDER_H
#define DATAPROVIDER_H
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>

#include "chunk.h"

using namespace std;

class DataProvider {
public:
    DataProvider(const string& images_file, const string& labels_file, bool shuffle = true);
    void load_batch(Chunk* images, Chunk* labels, int batch_size);
    inline int num_images() {return num_images_;};
private:
    void shuffle_data();
    vector<vector<float>> images_;
    vector<float> labels_;
    bool shuffle_;
    int index_in_epoch_;
    int num_images_;
};

#endif // DATAPROVIDER_H
