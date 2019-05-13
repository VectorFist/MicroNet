#ifndef GANNET2_H
#define GANNET2_H

#include "net.h"

namespace micronet {

class GanNet2: public Net {
public:
    GanNet2(const string& net_name="GanNet2");
    GanNet2(chunk_ptr (*generator_constructor)(const chunk_ptr&), int noise_dim,
           chunk_ptr (*discriminator_constructor)(const chunk_ptr&), const vector<int>& real_shape,
           const string& net_name="GanNet2");

    virtual void fit(const map<string, data_t>& train_data, const map<string, data_t>& valid_data,
                     int batch_size, int epochs, int verbose=100, bool shuffle=true) override;
    virtual void evaluate(const map<string, data_t>& data, int batch_size) override;
    virtual data_t inference(const map<string, data_t>& data, int batch_size) override;

protected:
    virtual void forward(bool is_train, const string& layer_prefix = "") override;
    virtual void backward(const string& layer_prefix = "") override;
    virtual void update(const string& layer_prefix = "") override;

    void forward_generator(bool is_train);
    void forward_discriminator_real(bool is_train);
    void forward_discriminator_fake(bool is_train);
    void save_generator_imgs(int iter);
};

} // namespace micronet

#endif // GANNET2_H
