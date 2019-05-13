#ifndef NET_H
#define NET_H
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <algorithm>
#include <iostream>
#include <utility>
#include <queue>
#include <iomanip>
#include <fstream>

#include "nlohmann/json.hpp"
#include "layer.h"
#include "chunk.h"
#include "sgdoptimizer.h"
#include "dataprovider.h"
#include "util.h"

using json = nlohmann::json;

namespace micronet {


using data_t = vector<vector<float>>;

extern map<string, vector<Layer*>> layer_space;

struct layer_compare {
    bool operator()(const layer_ptr& layer1, const layer_ptr& layer2) {
        return layer1.get() < layer2.get();
    }
};

void add_layer_prefix(const chunk_ptr& in, const chunk_ptr& out, const string& prefix);
void share_parameters(const chunk_ptr& in, const chunk_ptr& out, const string& layer_space_name);

class Net {
public:
    explicit Net(const string& net_name="Micronet");

    void set_optimizer(const shared_ptr<Optimizer>& optimizer);

    virtual void fit(const map<string, data_t>& train_data, const map<string, data_t>& valid_data,
                     int batch_size, int epochs, int verbose=100, bool shuffle=true) = 0;
    virtual void evaluate(const map<string, data_t>& data, int batch_size) = 0;
    virtual data_t inference(const map<string, data_t>& data, int batch_size) = 0;

    void save_model(const string& save_path);
    void load_model(const string& save_path);

    void print_net();

protected:
    void initialize();
    virtual void forward(bool is_train, const string& layer_prefix = "") = 0;
    virtual void backward(const string& layer_prefix = "") = 0;
    virtual void update(const string& layer_prefix = "") = 0;

    set<layer_ptr, layer_compare> all_layers_;
    map<layer_ptr, vector<layer_ptr>, layer_compare> net_graph_;
    vector<layer_ptr> net_sequences_;

    map<string, chunk_ptr> key_chunks_;
    vector<chunk_ptr> inputs_;
    shared_ptr<Optimizer> optimizer_;

    map<string, pair<double, double>> layer_op_time_;
    map<string, double> layer_up_time_;

    string net_name_;
    int iter_;

    bool net_initialized_ = false;

    friend void to_json(json& j_net, Net* net);
    friend void from_json(const json& j_net, Net* net);
};
} // namespace micronet

#endif // NET_H
