#ifndef ADAGRADOPTIMIZER_H
#define ADAGRADOPTIMIZER_H
#include <map>
#include "optimizer.h"

namespace micronet {

class AdaGradOptimizer: public Optimizer{
public:
    AdaGradOptimizer() = default;
    AdaGradOptimizer(float learning_rate, vector<float> decay_locs);
    virtual void optimize(const shared_ptr<Chunk>& param, int iter);
private:
    map<Chunk*, Chunk> accumulate_squared_gradient_;
};

} // namespace micronet

#endif // ADAGRADOPTIMIZER_H
