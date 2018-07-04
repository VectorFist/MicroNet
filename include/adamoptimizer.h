#ifndef ADAMOPTIMIZER_H
#define ADAMOPTIMIZER_H
#include <map>
#include "optimizer.h"


class AdamOptimizer: public Optimizer{
public:
    AdamOptimizer(float learning_rate, const vector<int>& decay_steps, float decay_rate_1 = 0.9, float decay_rate_2 = 0.999);
    virtual void optimize(shared_ptr<Chunk>& param, const string& param_name, int iter);
private:
    map<string, Chunk> first_moment_estimate_;
    map<string, Chunk> second_moment_estimate_;
    float decay_rate_1_, decay_rate_2_;
};

#endif // ADAMOPTIMIZER_H
