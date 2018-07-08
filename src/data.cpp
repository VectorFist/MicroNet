/**
 * @file data.cpp
 * @auther yefajie
 * @data 2018/6/22
 **/
#include "data.h"


Data::Data(const string& data_dir, int batch_size, const string& layer_name, const string& phase, bool shuffle) {
    layer_name_ = layer_name;
    batch_size_ = batch_size;
    if (phase == "train") {
        cout << "Read train data..." << endl;
        provider_ = make_shared<DataProvider>(data_dir+"/train-images.idx3-ubyte",
                                              data_dir+"/train-labels.idx1-ubyte", shuffle);
        cout << "Read train data done!!!" << endl << endl;
    } else {
        cout << "Read test data..." << endl;
        provider_ = make_shared<DataProvider>(data_dir+"/t10k-images.idx3-ubyte",
                                              data_dir+"/t10k-labels.idx1-ubyte", false);
        cout << "Read test data done!!!" << endl << endl;
    }
    cout << "Initialize data layer: " << layer_name_ << " done..." << endl;
}

void Data::set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks) {
    in_chunks_ = in_chunks;
    out_chunks_ = out_chunks;
}

void Data::forward(const vector<Chunk*>& input, const vector<Chunk*>& output) {
    provider_->load_batch(output[0], output[1], batch_size_);
}
