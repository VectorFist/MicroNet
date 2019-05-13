#ifndef RMSPROBOPTIMIZER_H
#define RMSPROBOPTIMIZER_H
#include <map>
#include <cmath>
#include "optimizer.h"

namespace micronet {

class RMSProbOptimizer: public Optimizer{
public:
    RMSProbOptimizer() = default;
    RMSProbOptimizer(float learning_rate, vector<float> decay_locs, float decay_rate = 0.5);
    virtual void optimize(const shared_ptr<Chunk>& param, int iter);
private:
    map<Chunk*, Chunk> accumulate_squared_gradient_;
};
} // namespace micronet

#endif // RMSPROBOPTIMIZER_H
