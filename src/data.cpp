/**
 * @file data.cpp
 * @auther yefajie
 * @data 2018/6/22
 **/
#include "data.h"

namespace micronet {

Data::Data(const string& data_dir, int batch_size, const string& layer_name, const string& phase, bool shuffle):
    Layer(layer_name, "Data"){
    batch_size_ = batch_size;
    if (phase == "train") {
        cout << "Read train data..." << endl;
        //provider_ = make_shared<DataProvider>(data_dir+"/train-images.idx3-ubyte",
        //                                      data_dir+"/train-labels.idx1-ubyte", shuffle);
        cout << "Read train data done!!!" << endl << endl;
    } else {
        cout << "Read test data..." << endl;
        //provider_ = make_shared<DataProvider>(data_dir+"/t10k-images.idx3-ubyte",
        //                                      data_dir+"/t10k-labels.idx1-ubyte", false);
        cout << "Read test data done!!!" << endl << endl;
    }
    cout << "Initialize data layer: " << layer_name_ << " done..." << endl;
}

} // namespace micronet
