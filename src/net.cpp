/**
 * @file net.cpp
 * @auther yefajie
 * @data 2018/6/26
 **/

#include "net.h"
#include "convolution.h"

namespace micronet {

map<string, vector<Layer*>> layer_space;

void add_layer_prefix(const chunk_ptr& in, const chunk_ptr& out, const string& prefix) {
    set<layer_ptr, layer_compare> layer_visited;
    queue<chunk_ptr> chunks;
    chunks.push(in);
    while (chunks.front().get() != out.get()) {
        chunk_ptr chk = chunks.front();
        chunks.pop();
        for (const auto& layer: chk->in_layers_) {
            if (layer_visited.find(layer) == layer_visited.end()) {
                layer->layer_name_ = prefix + "_" + layer->layer_name_;
                layer_visited.insert(layer);
                for (const auto& chunk: layer->chunks_out_) {
                    chunks.push(chunk);
                }
            }
        }
    }
}

void share_parameters(const chunk_ptr& in, const chunk_ptr& out, const string& layer_space_name) {
    vector<Layer*>& space = layer_space[layer_space_name];
    int index = 0;
    set<layer_ptr, layer_compare> layer_visited;
    queue<chunk_ptr> chunks;
    chunks.push(in);
    while (chunks.front().get() != out.get()) {
        chunk_ptr chk = chunks.front();
        chunks.pop();
        for (const auto& layer: chk->in_layers_) {
            if (layer_visited.find(layer) == layer_visited.end()) {
                layer_visited.insert(layer);
                for (const auto& chunk: layer->chunks_out_) {
                    chunks.push(chunk);
                }

                if (index == space.size()) {
                    space.push_back(layer.get());
                } else {
                    int num_params = layer->params_.size();
                    for (int n = 0; n < num_params; ++n) {
                        if (layer->params_[n]->trainable()) {
                            layer->params_[n] = space[index]->params_[n];
                            //cout << layer->params_[n].use_count() << "======================\n";
                        }
                    }
                }
                index++;
            }
        }
    }
}

Net::Net(const string& net_name): net_name_(net_name), iter_(0) {
}

void Net::set_optimizer(const shared_ptr<Optimizer>& optimizer) {
    optimizer_ = optimizer;
}

void Net::initialize() {
    queue<chunk_ptr> chunks;
    for (const auto& chunk: inputs_) {
        chunks.push(chunk);
    }
    while (!chunks.empty()) {
        chunk_ptr c = chunks.front();
        chunks.pop();
        cout << c->in_layers_.size() << endl;
        for (const auto& layer: c->in_layers_) {
            if (all_layers_.find(layer) == all_layers_.end()) {
                all_layers_.insert(layer);
                cout << layer->layer_name_ << ":" << layer->chunks_out_.size() <<endl;
                for (const auto& chunk: layer->chunks_out_) {
                    chunks.push(chunk);
                }
            }
        }
    }

    for (const auto& layer: all_layers_) {
        for (const auto& chunk: layer->chunks_out_) {
            if (chunk->in_layers_.empty()) {
                continue;
            }
            for (const auto& belong: chunk->in_layers_) {
                net_graph_[layer].push_back(belong);
            }
        }
    }

    layer_ptr guard_layer(nullptr);
    for (const auto& in_chunk: inputs_) {
        for (const auto& layer: in_chunk->in_layers_) {
            net_graph_[guard_layer].push_back(layer);
        }
    }

    for (const auto& from: net_graph_) {
        if (from.first.get() == nullptr) {
            cout << "guard: ";
        } else {
            cout << from.first->layer_name_ << ": ";
        }
        for (const auto& to: from.second) {
            cout << to->layer_name_ << ", ";
        }
        cout << endl;
    }

    map<layer_ptr, int, layer_compare> layer_inner_degree;
    for (const auto& edge: net_graph_) {
        for (const auto& out_node: edge.second) {
            layer_inner_degree[out_node] += 1;
        }
    }
    layer_inner_degree[guard_layer] = 0;
    for (const auto& layer: layer_inner_degree) {
        if (layer.first.get() == nullptr) {
            cout << "guard: " << layer.second << endl;
        } else {
            cout << layer.first->layer_name_ << ": " << layer.second << endl;
        }
    }

    while(net_sequences_.size() != all_layers_.size() + 1) {
        int size_flag = layer_inner_degree.size();
        for (const auto& layer: layer_inner_degree) {
            if (layer.second == 0) {
                net_sequences_.push_back(layer.first);
                for (const auto& point_layer: net_graph_[layer.first]) {
                    layer_inner_degree[point_layer] -= 1;
                }
                layer_inner_degree.erase(layer.first);
            }
        }
        if (layer_inner_degree.size() == size_flag) {
            cout << "the graph built must be undirected..." << endl;
            exit(1);
        }
    }
    net_sequences_.erase(net_sequences_.begin());

    for (const auto& layer: net_sequences_) {
        cout << layer->layer_name_ << " --> ";
        for (const auto& point_layer: net_graph_[layer]) {
            cout << point_layer->layer_name_ << ", ";
        }
        cout << endl;
    }

    net_initialized_ = true;
    cout << "Initialize net done !" << endl << endl;
}

void Net::save_model(const string& save_path) {
    Timer timer;

    json j_net;
    to_json(j_net, this);
    std::ofstream ofs(save_path);
    ofs << j_net << endl;

    cout << "save model use time: " << timer.elapsed() << " s" << endl;
}

void Net::load_model(const string& save_path) {
    Timer timer;

    json j_net;
    std::ifstream ifs(save_path);
    ifs >> j_net;
    from_json(j_net, this);

    net_initialized_ = true;
    cout << "load model use time: " << timer.elapsed() << " s" << endl;
}

void Net::print_net() {
    for (const auto& layer: net_sequences_) {
        cout << layer->layer_type_ << ": " << setw(20) << layer->layer_name_ << endl;
        cout << layer->chunks_in_.size() << " " << layer->chunks_out_.size() << endl;
        for (const auto& chunk: layer->chunks_in_) {
            cout << chunk->str_shape() << endl;
        }
        for (const auto& chunk: layer->chunks_out_) {
            cout << chunk->str_shape() << endl;
        }
    }
}

} // namespace micronet
