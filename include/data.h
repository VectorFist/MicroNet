#ifndef DATA_H
#define DATA_H
#include <memory>
#include "layer.h"
#include "dataprovider.h"

namespace micronet {

class Data: public Layer {
public:
    Data(const string& data_dir, int batch_size,  const string& layer_name, const string& phase, bool shuffle = true);
    virtual void forward(bool is_train=true) override {};
    virtual void backward() override {};

protected:
    virtual vector<int> shape_inference() override {};

private:
    shared_ptr<DataProvider> provider_;
    int batch_size_;
};
} // namespace micornet

#endif // DATA_H
