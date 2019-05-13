#ifndef ADAMOPTIMIZER_H
#define ADAMOPTIMIZER_H
#include <map>
#include "optimizer.h"

namespace micronet {

class AdamOptimizer: public Optimizer{
public:
    AdamOptimizer() = default;
    AdamOptimizer(float learning_rate, vector<float> decay_locs, float decay_rate_1 = 0.9, float decay_rate_2 = 0.999);
    virtual void optimize(const shared_ptr<Chunk>& param, int iter);

private:
    map<Chunk*, Chunk> first_moment_estimate_;
    map<Chunk*, Chunk> second_moment_estimate_;
};
} // namespace micronet

#endif // ADAMOPTIMIZER_H
