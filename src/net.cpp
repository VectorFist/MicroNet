/**
 * @file net.cpp
 * @auther yefajie
 * @data 2018/6/26
 **/
#include <sstream>
#include "net.h"

Net::Net(const string& net_name): net_name_(net_name), iter_(0) {
}

void Net::add_layer(const shared_ptr<Layer>& layer) {
    if (all_layers_.find(layer->layer_name_) != all_layers_.end()) {
        cout << "Add layer error: " << layer->layer_name_ << " exists, please rename it!";
        exit(1);
    }
    all_layers_[layer->layer_name_] = layer;
    for (const string& chunk: layer->in_chunks_) {
        if (all_chunks_.find(chunk) == all_chunks_.end()) {
            all_chunks_[chunk] = Chunk();
        }
        input_chunks_belongs_[chunk].push_back(layer->layer_name_);
    }
    for (const string& chunk: layer->out_chunks_) {
        if (all_chunks_.find(chunk) == all_chunks_.end()) {
            all_chunks_[chunk] = Chunk();
        }
    }
}

void Net::set_data_layers(const vector<string>& data_layers) {
    data_layers_ = data_layers;
}

void Net::set_default_data_layer(const string& data_layer) {
    if (!is_initialized_) {
        cout << "Net initialize error: please initialize net first before set default data layer!";
        exit(1);
    }
    for (const string& layer: data_layers_) {
        if (find(net_sequences_.begin(), net_sequences_.end(), layer) != net_sequences_.end()) {
            net_sequences_.erase(find(net_sequences_.begin(), net_sequences_.end(), layer));
        }
    }
    net_sequences_.insert(net_sequences_.begin(), data_layer);
}

void Net::set_optimizer(const shared_ptr<Optimizer>& optimizer) {
    optimizer_ = optimizer;
}

void Net::initialize() {
    for (const auto& layer: all_layers_) {
        const vector<string>& out_chunks = layer.second->out_chunks_;
        const string& layer_name = layer.first;
        for (const string& chunk: out_chunks) {
            if (input_chunks_belongs_.find(chunk) == input_chunks_belongs_.end()) {
                continue;
            }
            const vector<string>& belongs = input_chunks_belongs_[chunk];
            for (const string& belong: belongs) {
                net_graph_[layer_name].push_back(belong);
            }
        }
    }

    map<string, int> layer_inner_degree;
    for (const auto& edge: net_graph_) {
        layer_inner_degree[edge.first] = 0;
    }
    for (const auto& edge: net_graph_) {
        for (const string& out_node: edge.second) {
            layer_inner_degree[out_node] += 1;
        }
    }

    while(net_sequences_.size() != all_layers_.size()) {
        for (const auto& layer: layer_inner_degree) {
            if (layer.second == 0) {
                net_sequences_.push_back(layer.first);
                layer_inner_degree.erase(layer.first);
                for (const string& point_layer: net_graph_[layer.first]) {
                    layer_inner_degree[point_layer] -= 1;
                }
            }
        }
    }

    cout << "Initialize net ..." << endl;
    for (const string& layer: net_sequences_) {
        cout << layer << " --> ";
        for (const string& point_layer: net_graph_[layer]) {
            cout << point_layer << ", ";
        }
        cout << endl;
    }
    cout << "Initialize net done !" << endl << endl;
    is_initialized_ = true;
}

void Net::forward() {
    if (!is_initialized_) {
        cout << "Net initialize error: please initialize net first before net forward!";
        exit(1);
    }

    for (auto layer = net_sequences_.begin(); layer != net_sequences_.end(); ++layer) {
        vector<Chunk*> input, output;
        for (const string& chunk: all_layers_[*layer]->in_chunks_) {
            input.push_back(&all_chunks_[chunk]);
        }
        for (const string& chunk: all_layers_[*layer]->out_chunks_) {
            output.push_back(&all_chunks_[chunk]);
        }
        all_layers_[*layer]->forward(input, output);
    }
}


void Net::backward() {
    for (auto layer = net_sequences_.rbegin(); layer != net_sequences_.rend(); ++layer) {
        vector<Chunk*> input, output;
        for (const string& chunk: all_layers_[*layer]->in_chunks_) {
            input.push_back(&all_chunks_[chunk]);
        }
        for (const string& chunk: all_layers_[*layer]->out_chunks_) {
            output.push_back(&all_chunks_[chunk]);
        }
        all_layers_[*layer]->backward(input, output);
    }
}

void Net::update() {
    iter_++;
    for (const string& layer: net_sequences_) {
        vector<shared_ptr<Chunk>>& params = all_layers_[layer]->params_;
        for (unsigned i = 0; i < params.size(); ++i) {
            stringstream param_name;
            param_name << layer << "_" << i;
            optimizer_->optimize(params[i], param_name.str(), iter_);
        }
    }
}
