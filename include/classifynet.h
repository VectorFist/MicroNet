#ifndef CLASSIFYNET_H
#define CLASSIFYNET_H

#include "net.h"
#include "softmaxloss.h"
#include "accuracy.h"

namespace micronet {

class ClassifyNet: public Net {
public:
    explicit ClassifyNet(const string& net_name="ClassifyNet");
    explicit ClassifyNet(chunk_ptr (*graph_constructor)(const chunk_ptr&), const vector<int>& img_shape,
                         const string& net_name="ClassifyNet");

    virtual void fit(const map<string, data_t>& train_data, const map<string, data_t>& valid_data,
                     int batch_size, int epochs, int verbose=100, bool shuffle=true) override;
    virtual void evaluate(const map<string, data_t>& data, int batch_size) override;
    virtual data_t inference(const map<string, data_t>& data, int batch_size) override;

protected:
    virtual void forward(bool is_train, const string& layer_prefix = "") override;
    virtual void backward(const string& layer_prefix = "") override;
    virtual void update(const string& layer_prefix = "") override;
};

} // namespace micronet

#endif // CLASSIFYNET_H
