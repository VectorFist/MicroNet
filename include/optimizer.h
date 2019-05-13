#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <memory>
#include <cmath>
#include <map>
#include <algorithm>

#include "nlohmann/json.hpp"
#include "chunk.h"

using json = nlohmann::json;

namespace micronet {

class Net;

class Optimizer {
public:
    Optimizer() = default;
    Optimizer(const string& optimizer_type, vector<float> decay_locs);
    virtual ~Optimizer(){};
    virtual void optimize(const shared_ptr<Chunk>& param, int iter) = 0;
    int total_iters_;

protected:
    string optimizer_type_;
    vector<float> decay_locs_;
    map<string, string> str_hps_;
    map<string, float> flt_hps_;
    map<string, int> int_hps_;

    friend class Net;

    friend shared_ptr<Optimizer> parse_optimizer(const json& j_optimizer);
    friend void to_json(json& j_net, Net* net);
};
} // namespace micronet

#endif // OPTIMIZER_H
