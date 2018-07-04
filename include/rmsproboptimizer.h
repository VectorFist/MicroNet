#ifndef RMSPROBOPTIMIZER_H
#define RMSPROBOPTIMIZER_H
#include <map>
#include <cmath>
#include "optimizer.h"

class RMSProbOptimizer: public Optimizer{
public:
    RMSProbOptimizer(float learning_rate, const vector<int>& decay_steps, float decay_rate = 0.5);
    virtual void optimize(shared_ptr<Chunk>& param, const string& param_name, int iter);
private:
    map<string, Chunk> accumulate_squared_gradient_;
    float decay_rate_;
};

#endif // RMSPROBOPTIMIZER_H
