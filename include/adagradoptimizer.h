#ifndef ADAGRADOPTIMIZER_H
#define ADAGRADOPTIMIZER_H
#include <map>
#include "optimizer.h"

class AdaGradOptimizer: public Optimizer{
public:
    AdaGradOptimizer(float learning_rate, const vector<int>& decay_steps);
    virtual void optimize(shared_ptr<Chunk>& param, const string& param_name, int iter);
private:
    map<string, Chunk> accumulate_squared_gradient_;
};

#endif // ADAGRADOPTIMIZER_H
