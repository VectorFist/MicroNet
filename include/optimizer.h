#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <memory>
#include <cmath>
#include <algorithm>
#include "chunk.h"

class Optimizer {
public:
    Optimizer(float learning_rate, const vector<int>& decay_steps);
    virtual ~Optimizer(){};
    virtual void optimize(shared_ptr<Chunk>& param, const string& param_name, int iter) = 0;
protected:
    float learning_rate_;
    vector<int> decay_steps_;
};

#endif // OPTIMIZER_H
