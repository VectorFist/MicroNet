#ifndef DATA_H
#define DATA_H
#include <memory>
#include "layer.h"
#include "dataprovider.h"


class Data: public Layer {
public:
    Data(const string& data_dir, int batch_size,  const string& layer_name, const string& phase, bool shuffle = true);
    virtual void set_chunks(const vector<string>& in_chunks, const vector<string>& out_chunks);
    virtual void forward(const vector<Chunk*>& input, const vector<Chunk*>& output);
    virtual void backward(const vector<Chunk*>& input, const vector<Chunk*>& output){};
private:
    shared_ptr<DataProvider> provider_;
    int batch_size_;
};

#endif // DATA_H
