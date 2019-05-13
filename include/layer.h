#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <map>

#include "nlohmann/json.hpp"
#include "chunk.h"

using json = nlohmann::json;

#define FLT_MAX 3.402823466e+30F
#define FLT_MIN 1.175494351e-30F

using namespace std;

namespace micronet {

using chunk_ptr = shared_ptr<Chunk>;

class Net;

class Layer {
public:
    Layer() = default;
    Layer(const string& layer_name, const string& layer_type);
    virtual ~Layer(){};
    virtual void forward(bool is_train=true) = 0;
    virtual void backward() = 0;

protected:
    virtual vector<int> shape_inference() = 0;
    void gradient_reset();
    vector<chunk_ptr> params_;
    vector<chunk_ptr> chunks_in_, chunks_out_;
    string layer_name_;
    string layer_type_;

    map<string, string> str_hps_;  //layer string hyper params
    map<string, float> flt_hps_;   //layer float hyper params
    map<string, int> int_hps_; //layer int hyper params

    friend class Net;
    friend class ClassifyNet;
    friend class GanNet;
    friend class GanNet2;
    friend class RegressionNet;
    friend shared_ptr<Layer> parse_layer(const json& j_layer, map<string, shared_ptr<Chunk>>& params);
    friend void to_json(json& j_net, Net* net);
    friend void from_json(const json& j_net, Net* net);
    friend void add_layer_prefix(const chunk_ptr& in, const chunk_ptr& out, const string& suffix);
    friend void add_layer_prefix2(const chunk_ptr& in, const chunk_ptr& out, const string& suffix);
    friend void share_parameters(const chunk_ptr& in, const chunk_ptr& out, const string& layer_space_name);
};
} // namespae micronet

#endif // F:\DDisk\CodeBlocks\MicroNet\include\layer.hLAYER_H
