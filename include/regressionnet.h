#ifndef REGRESSIONNET_H
#define REGRESSIONNET_H

#include "net.h"
#include "l2loss.h"


namespace micronet {

class RegressionNet: public Net {
public:
    explicit RegressionNet(const string& net_name="RegressionNet");
    explicit RegressionNet(chunk_ptr (*graph_constructor)(const vector<chunk_ptr>&),
                           const vector<vector<int>>& input_shapes,
                           const string& net_name="RegressionNet");

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

#endif // REGRESSIONNET_H
