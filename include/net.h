#ifndef NET_H
#define NET_H
#include <vector>
#include <memory>
#include <map>
#include <algorithm>
#include <iostream>
#include "layer.h"
#include "chunk.h"
#include "sgdoptimizer.h"


class Net {
public:
    Net(const string& net_name);
    void add_layer(const shared_ptr<Layer>& layer);
    void set_data_layers(const vector<string>& data_layer);
    void set_default_data_layer(const string& data_layer);
    void set_optimizer(const shared_ptr<Optimizer>& optimizer);

    void initialize();
    void forward();
    void backward();
    void update();

    map<string, shared_ptr<Layer>> all_layers_;
    map<string, Chunk> all_chunks_;
    map<string, vector<string>> net_graph_;
    map<string, vector<string>> input_chunks_belongs_;

    vector<string> net_sequences_;
    string net_name_;
    vector<string> data_layers_;
    bool is_initialized_ = false;
    int iter_;

    shared_ptr<Optimizer> optimizer_;
};

#endif // NET_H
