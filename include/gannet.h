#ifndef GANNET_H
#define GANNET_H

#include "net.h"
#include "softmaxloss.h"
#include "sigmoidloss.h"
#include "concatenate.h"
#include "batchmiddlesplit.h"
#include "util.h"


namespace micronet {

class GanNet: public Net {
public:
    GanNet(const string& net_name="GanNet");
    GanNet(chunk_ptr (*generator_constructor)(const chunk_ptr&), int noise_dim,
           chunk_ptr (*discriminator_constructor)(const chunk_ptr&), const vector<int>& real_shape,
           const string& net_name="GanNet");

    virtual void fit(const map<string, data_t>& train_data, const map<string, data_t>& valid_data,
                     int batch_size, int epochs, int verbose=100, bool shuffle=true) override;
    virtual void evaluate(const map<string, data_t>& data, int batch_size) override;
    virtual data_t inference(const map<string, data_t>& data, int batch_size) override;

protected:
    virtual void forward(bool is_train, const string& layer_prefix = "") override;
    virtual void backward(const string& layer_prefix = "") override;
    virtual void update(const string& layer_prefix = "") override;

    void forward_generator(bool is_train);
    void forward_discriminator(bool is_train);
    void save_generator_imgs(int iter);
};

} // namespace micronet

#endif // GANNET_H
