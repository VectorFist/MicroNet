#ifndef SGDOPTIMIZER_H
#define SGDOPTIMIZER_H
#include <map>
#include "optimizer.h"

namespace micronet {

class SGDOptimizer: public Optimizer {
public:
    SGDOptimizer() = default;
    SGDOptimizer(float learning_rate, vector<float> decay_locs, float momentum = 0.9);
    virtual void optimize(const shared_ptr<Chunk>& param, int iter);
private:
    map<Chunk*, Chunk> param_velocity_;
};
} // namespace micronet

#endif // SGDOPTIMIZER_H
