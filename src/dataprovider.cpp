/**
 * @file dataprovider.cpp
 * @auther yefajie
 * @data 2018/6/22
 **/
#include "dataprovider.h"
#include "util.h"

DataProvider::DataProvider(const string& images_file, const string& labels_file, bool shuffle) {
    read_mnist_images(images_file, images_);
    read_mnist_lables(labels_file, labels_);
    shuffle_ = shuffle;
    index_in_epoch_ = 0;
    num_images_ = images_.size();
    if (shuffle_) {
        shuffle_data();
    }
}

void DataProvider::load_batch(Chunk* images, Chunk* labels, int batch_size) {
    if (index_in_epoch_ + batch_size > num_images_) {
        if (shuffle_) {
            shuffle_data();
        }
        index_in_epoch_ = 0;
    }
    int load_start = index_in_epoch_;
    int load_end = load_start + batch_size;
    index_in_epoch_ += batch_size;

    int num = batch_size;
    int channels = 1;
    int width = 28;
    int height = 28;
    int dim = 28 * 28;
    images->reshape(num, channels, height, width);
    labels->reshape(num, 1, 1, 1);

    float* images_data = images->data();
    float* labels_data = labels->data();
    for (int n = load_start; n < load_end; ++n) {
        for (int d = 0; d < dim; ++d) {
            images_data[d] = images_[n][d] / 255.0;
        }
        labels_data[n-load_start] = labels_[n];
        images_data += dim;
    }
}

void DataProvider::shuffle_data() {
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(images_.begin(), images_.end(), default_random_engine(seed));
    shuffle(labels_.begin(), labels_.end(), default_random_engine(seed));
    cout << "Shuffle data done, load batch data from a new epoch..." << endl;
}
