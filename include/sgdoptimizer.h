#ifndef SGDOPTIMIZER_H
#define SGDOPTIMIZER_H
#include <map>
#include "optimizer.h"

class SGDOptimizer: public Optimizer {
public:
    SGDOptimizer(float learning_rate, const vector<int>& decay_steps, float momentum = 0.9);
    virtual void optimize(shared_ptr<Chunk>& param, const string& param_name, int iter);
private:
    map<string, Chunk> param_velocity_;
    float momentum_;
};

#endif // SGDOPTIMIZER_H
