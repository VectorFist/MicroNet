/**
 * @file util.cpp
 * @auther yefajie
 * @data 2018/6/22
 **/
#include <omp.h>

#include "adagradoptimizer.h"
#include "adamoptimizer.h"
#include "rmsproboptimizer.h"
#include "sgdoptimizer.h"
#include "accuracy.h"
#include "activation.h"
#include "add.h"
#include "argmax.h"
#include "convolution.h"
#include "concatenate.h"
#include "dense.h"
#include "focalloss.h"
#include "pooling.h"
#include "softmax.h"
#include "softmaxloss.h"
#include "sigmoidloss.h"
#include "util.h"
#include "deconvolution.h"
#include "l2loss.h"
#include "croppingimage.h"
#include "paddingimage.h"
#include "dropout.h"
#include "batchnormalization.h"
#include "reshape.h"
#include "pixelshuffle.h"
#include "instancenormalization.h"

namespace micronet {

static unsigned global_seed = std::chrono::system_clock::now().time_since_epoch().count();

int ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_mnist_lables(const string& filename, vector<vector<float>>& labels) {
    ifstream file(filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        cout << "magic number = " << magic_number << endl;
        cout << "number of images = " << number_of_images << endl;

        for (int i = 0; i < number_of_images; i++)
        {
            unsigned char label = 0;
            file.read((char*)&label, sizeof(label));
            labels.push_back({(float)label});
        }
    }
}

void read_mnist_images(const string& filename, vector<vector<float>>& images) {
    ifstream file(filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        file.read((char*)&n_rows, sizeof(n_rows));
        file.read((char*)&n_cols, sizeof(n_cols));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        n_rows = ReverseInt(n_rows);
        n_cols = ReverseInt(n_cols);

        cout << "magic number = " << magic_number << endl;
        cout << "number of images = " << number_of_images << endl;
        cout << "rows = " << n_rows << endl;
        cout << "cols = " << n_cols << endl;

        for (int i = 0; i < number_of_images; i++)
        {
            vector<float> tp;
            for (int r = 0; r < n_rows; r++)
            {
                for (int c = 0; c < n_cols; c++)
                {
                    unsigned char image = 0;
                    file.read((char*)&image, sizeof(image));
                    tp.push_back((float)image);
                }
            }
            images.push_back(tp);
        }
    }
}

void read_cifar10_images(const string& dirname, vector<vector<float>>& images, bool train) {
    if (!train) {
        string filename = dirname;
        ifstream ifs(filename, ios::binary);
        unsigned char pixel;
        for (int n = 0; n < 10000; ++n) {
            ifs.read((char*)&pixel, sizeof(pixel));
            vector<float> image;
            for (int j = 0; j < 1024*3; ++j) {
                ifs.read((char*)&pixel, sizeof(pixel));
                image.push_back((float)pixel);
            }
            images.push_back(image);
        }
        return;
    }
    for (int i = 1; i <= 4; ++i) {
        string filename = dirname + "/data_batch_" + to_string(i) + ".bin";
        ifstream ifs(filename, ios::binary);
        unsigned char pixel;
        for (int n = 0; n < 10000; ++n) {
            ifs.read((char*)&pixel, sizeof(pixel));
            vector<float> image;
            for (int j = 0; j < 1024*3; ++j) {
                ifs.read((char*)&pixel, sizeof(pixel));
                image.push_back((float)pixel);
            }
            images.push_back(image);
        }
    }
}

void read_cifar10_gan_imgs(const string& dirname, vector<vector<float>>& images, int category) {
    string filename = dirname + "/test_batch.bin";
    ifstream ifs(filename, ios::binary);
    unsigned char pixel, type;
    for (int n = 0; n < 10000; ++n) {
        ifs.read((char*)&type, sizeof(type));
        vector<float> image;
        for (int j = 0; j < 1024*3; ++j) {
            ifs.read((char*)&pixel, sizeof(pixel));
            image.push_back((float)pixel);
        }
        if (type == category) {
            images.push_back(image);
        }
    }
    for (int i = 1; i <= 5; ++i) {
        string filename = dirname + "/data_batch_" + to_string(i) + ".bin";
        ifstream ifs(filename, ios::binary);
        unsigned char pixel, type;
        for (int n = 0; n < 10000; ++n) {
            ifs.read((char*)&type, sizeof(type));
            vector<float> image;
            for (int j = 0; j < 1024*3; ++j) {
                ifs.read((char*)&pixel, sizeof(pixel));
                image.push_back((float)pixel);
            }
            if (type == category) {
                images.push_back(image);
            }
        }
    }
}

void read_cifar100_images(const string& dirname, vector<vector<float>>& images) {
    string filename = dirname + "/train.bin";
    ifstream ifs(filename, ios::binary);
    unsigned char pixel;
    for (int n = 0; n < 50000; ++n) {
        ifs.read((char*)&pixel, sizeof(pixel));
        ifs.read((char*)&pixel, sizeof(pixel));
        vector<float> image;
        for (int j = 0; j < 1024*3; ++j) {
            ifs.read((char*)&pixel, sizeof(pixel));
            image.push_back((float)pixel);
        }
        images.push_back(image);
    }
}

void cal_low_imgs(const vector<vector<float>>& images, vector<vector<float>>& low_images, float factor) {
    for (int n = 0; n < images.size(); ++n) {
        float rate = factor;
        vector<float> down_sampled_img(int(3*32*rate*32*rate));
        bilinear_interpolation(32, 32, 3, images[n].data(), rate, rate, down_sampled_img.data());
        rate = 1 / factor;
        vector<float> up_sampled_img(int(3*32*factor*rate*32*factor*rate));
        bilinear_interpolation(int(32*factor), int(32*factor), 3, down_sampled_img.data(), rate, rate, up_sampled_img.data());
        low_images.push_back(up_sampled_img);
    }

}

void cal_L_component_from_RGB(const vector<vector<float>>& rgb_images, vector<vector<float>>& l_images) {
    int img_size = rgb_images[0].size() / 3;
    for (int n = 0; n < rgb_images.size(); ++n) {
        vector<float> l_img;
        float max_pixel = 255, min_pixel = 0;
        for (int i = 0; i < img_size; ++i) {
            max_pixel = max(max(rgb_images[n][i], rgb_images[n][i+img_size]), rgb_images[n][i+img_size+img_size]);
            min_pixel = min(min(rgb_images[n][i], rgb_images[n][i+img_size]), rgb_images[n][i+img_size+img_size]);
            l_img.push_back((max_pixel+min_pixel)/2.0);
        }
        l_images.push_back(l_img);
    }
}


float img2col_get_pixel(const float* data_im, int height, int width, int channels,
                       int row, int col, int channel, int pad_h, int pad_w) {
    row -= pad_h;
    col -= pad_w;
    if (row < 0 || row >= height || col < 0 || col >= width) {
        return 0;
    }
    return data_im[(height * channel + row) * width + col];
}


void img2col(const float* data_im, int channels,  int height,  int width, int ksize_h,
            int ksize_w, int pad_h, int pad_w, int stride_h, int stride_w, float* data_col) {
    int c,h,w;
    int height_col = (height + 2*pad_h - ksize_h) / stride_h + 1;
    int width_col = (width + 2*pad_w - ksize_w) / stride_w + 1;

    int channels_col = channels * ksize_h * ksize_w;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize_w;
        int h_offset = (c / ksize_w) % ksize_h;
        int c_im = c / ksize_h / ksize_w;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride_h;
                int im_col = w_offset + w * stride_w;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = img2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad_h, pad_w);
            }
        }
    }
}

void col2im_add_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad_h, int pad_w, float val) {
    row -= pad_h;
    col -= pad_w;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return;
    im[col + width*(row + height*channel)] += val;
}

void col2img(const float* data_col, int channels, int height, int width, int ksize_h,
             int ksize_w, int pad_h, int pad_w, int stride_h, int stride_w, float* data_im) {
    int c,h,w;
    int height_col = (height + 2*pad_h - ksize_h) / stride_h + 1;
    int width_col = (width + 2*pad_w - ksize_w) / stride_w + 1;

    int channels_col = channels * ksize_h * ksize_w;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize_w;
        int h_offset = (c / ksize_w) % ksize_h;
        int c_im = c / ksize_h / ksize_w;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride_h;
                int im_col = w_offset + w * stride_w;
                int col_index = (c * height_col + h) * width_col + w;
                double val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad_h, pad_w, val);
            }
        }
    }
}

void normal_random_init(int n, float* x, float mean, float stddev, int seed) {
    if (seed == -1) {
        global_seed += 1001;
        seed = global_seed;
    }
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(mean, stddev);
    for (int i = 0; i < n; ++i) {
        float rnd = distribution(generator);
        int trial = 0;
        while ((rnd < (mean - 2 * stddev) ||
                rnd > (mean + 2 * stddev)) && trial < 5) {
            rnd = distribution(generator);
            trial++;
        }
        x[i] = rnd;
    }
}

void uniform_random_init(int n, float* x, float lower, float upper, int seed) {
    if (seed == -1) {
        global_seed += 1001;
        seed = global_seed;
    }
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<float> distribution(lower, upper);
    for (int i = 0; i < n; ++i) {
        x[i] = distribution(generator);
    }
}

void constant_init(int n, float* x, float val) {
    for (int i = 0; i < n; ++i) {
        x[i] = val;
    }
}

shared_ptr<Optimizer> parse_optimizer(const json& j_optimizer) {
    string optimizer_type = j_optimizer["optimizer_type"].get<string>();
    shared_ptr<Optimizer> optimizer;
    if (optimizer_type == "AdaGrad") {
        optimizer = make_shared<AdaGradOptimizer>();
    } else if (optimizer_type == "Adam") {
        optimizer = make_shared<AdamOptimizer>();
    } else if (optimizer_type == "RMSProb") {
        optimizer = make_shared<RMSProbOptimizer>();
    } else if (optimizer_type == "SGD") {
        optimizer = make_shared<SGDOptimizer>();
    }
    optimizer->optimizer_type_ = optimizer_type;
    optimizer->decay_locs_ = j_optimizer["decay_locs"].get<vector<float>>();
    optimizer->str_hps_ = j_optimizer["str_hps"].get<map<string, string>>();
    optimizer->flt_hps_ = j_optimizer["flt_hps"].get<map<string, float>>();
    optimizer->int_hps_ = j_optimizer["int_hps"].get<map<string, int>>();

    return optimizer;
}

shared_ptr<Chunk> parse_chunk(const json& j_chunk) {
    auto shape = j_chunk["shape"].get<vector<int>>();
    shared_ptr<Chunk> chunk = make_shared<Chunk>(shape);

    return chunk;
}

shared_ptr<Chunk> parse_param(const json& j_param, map<string, shared_ptr<Chunk>>& params) {
    auto param_id = j_param["param_id"].get<string>();
    if (params.find(param_id) != params.end()) {
        return params[param_id];
    }
    auto shape = j_param["shape"].get<vector<int>>();
    auto data = j_param["data"].get<vector<float>>();
    shared_ptr<Chunk> param = make_shared<Chunk>(shape);
    //param->data_ = make_shared<vector<float>>(data);
    param->data_ = new float[data.size()];
    std::copy(data.begin(), data.end(), param->data_);
    param->trainable_ = j_param["trainable"].get<bool>();

    params[param_id] = param;

    return param;
}

shared_ptr<Layer> parse_layer(const json& j_layer, map<string, shared_ptr<Chunk>>& params) {
    string layer_type = j_layer["layer_type"].get<string>();
    shared_ptr<Layer> layer;
    if (layer_type == "Accuracy") {
        layer = make_shared<Accuracy>();
    } else if (layer_type == "Activation") {
        layer = make_shared<Activation>();
    } else if (layer_type == "Add") {
        layer = make_shared<Add>();
    } else if (layer_type == "ArgMax") {
        layer = make_shared<ArgMax>();
    } else if (layer_type == "Convolution") {
        layer = make_shared<Convolution>();
    } else if (layer_type == "Concatenate") {
        layer = make_shared<Concatenate>();
    } else if (layer_type == "Dense") {
        layer = make_shared<Dense>();
    } else if (layer_type == "FocalLoss") {
        layer = make_shared<FocalLoss>();
    } else if (layer_type == "Pooling") {
        layer = make_shared<Pooling>();
    } else if (layer_type == "Dense") {
        layer = make_shared<Dense>();
    } else if (layer_type == "Softmax") {
        layer = make_shared<Softmax>();
    } else if (layer_type == "SoftmaxLoss") {
        layer = make_shared<SoftmaxLoss>();
    } else if (layer_type == "SigmoidLoss") {
        layer = make_shared<SigmoidLoss>();
    } else if (layer_type == "Deconvolution") {
        layer = make_shared<Deconvolution>();
    } else if (layer_type == "L2Loss") {
        layer = make_shared<L2Loss>();
    } else if (layer_type == "CroppingImage") {
        layer = make_shared<CroppingImage>();
    } else if (layer_type == "PaddingImage") {
        layer = make_shared<PaddingImage>();
    } else if (layer_type == "Dropout") {
        layer = make_shared<Dropout>();
    } else if (layer_type == "BatchNormalization") {
        layer = make_shared<BatchNormalization>();
    } else if (layer_type == "Reshape") {
        layer = make_shared<Reshape>();
    } else if (layer_type == "PixelShuffle") {
        layer = make_shared<PixelShuffle>();
    } else if (layer_type == "InstanceNormalization") {
        layer = make_shared<InstanceNormalization>();
    }
    layer->layer_name_ = j_layer["layer_name"].get<string>();
    layer->layer_type_ = layer_type;
    layer->str_hps_ = j_layer["str_hps"].get<map<string, string>>();
    layer->flt_hps_ = j_layer["flt_hps"].get<map<string, float>>();
    layer->int_hps_ = j_layer["int_hps"].get<map<string, int>>();

    for (const json& j_param: j_layer["params"]) {
        shared_ptr<Chunk> param = parse_param(j_param, params);
        layer->params_.push_back(param);
    }

    return layer;
}

void to_json(json& j_net, Net* net) {
    j_net["net_name"] = net->net_name_;
    j_net["iter"] = net->iter_;
    j_net["optimizer"]["optimizer_type"] = net->optimizer_->optimizer_type_;
    j_net["optimizer"]["decay_locs"] = net->optimizer_->decay_locs_;
    j_net["optimizer"]["str_hps"] = net->optimizer_->str_hps_;
    j_net["optimizer"]["flt_hps"] = net->optimizer_->flt_hps_;
    j_net["optimizer"]["int_hps"] = net->optimizer_->int_hps_;

    j_net["inputs"] = {};
    for (const auto& chunk: net->inputs_) {
        j_net["inputs"].push_back(to_string(long(chunk.get())));
    }
    for (const auto& key_chunk: net->key_chunks_) {
        j_net["key_chunks"][key_chunk.first] = to_string(long(key_chunk.second.get()));
    }

    j_net["layers"] = {};
    for (const auto& layer: net->net_sequences_) {
        json j_layer;
        j_layer["layer_name"] = layer->layer_name_;
        j_layer["layer_type"] = layer->layer_type_;
        j_layer["layer_id"] = to_string(long(layer.get()));
        j_layer["to_layers"] = {};
        for (const auto& to: net->net_graph_[layer]) {
            j_layer["to_layers"].push_back(to_string(long(to.get())));
        }
        j_layer["str_hps"] = layer->str_hps_;
        j_layer["flt_hps"] = layer->flt_hps_;
        j_layer["int_hps"] = layer->int_hps_;
        j_layer["params"] = {};
        for (const auto& param: layer->params_) {
            json j_param;
            j_param["param_id"] = to_string(long(param.get()));
            j_param["shape"] = param->shape();
            j_param["trainable"] = param->trainable();
            j_param["data"] = vector<float>(param->const_data(), param->const_data()+param->count());;
            j_layer["params"].push_back(j_param);
        }
        j_layer["chunks_in"] = {};
        for (const auto& chunk: layer->chunks_in_) {
            json j_chunk;
            j_chunk["chunk_id"] = to_string(long(chunk.get()));
            j_chunk["shape"] = {1, chunk->channels(), chunk->height(), chunk->width()};
            j_layer["chunks_in"].push_back(j_chunk);
        }
        j_layer["chunks_out"] = {};
        for (const auto& chunk: layer->chunks_out_) {
            json j_chunk;
            j_chunk["chunk_id"] = to_string(long(chunk.get()));
            j_chunk["shape"] = {1, chunk->channels(), chunk->height(), chunk->width()};
            j_layer["chunks_out"].push_back(j_chunk);
        }
        j_net["layers"].push_back(j_layer);
    }
}

void from_json(const json& j_net, Net* net) {
    net->net_name_ = j_net["net_name"].get<string>();
    net->iter_ = j_net["iter"].get<int>();
    net->optimizer_ = parse_optimizer(j_net["optimizer"]);

    map<string, shared_ptr<Layer>> layers;
    map<string, shared_ptr<Chunk>> chunks;
    map<string, shared_ptr<Chunk>> params;
    for (const json& j_layer: j_net["layers"]) {
        string layer_id = j_layer["layer_id"].get<string>();
        layers[layer_id] = parse_layer(j_layer, params);
        for (const json& j_chunk: j_layer["chunks_in"]) {
            string chunk_id = j_chunk["chunk_id"].get<string>();
            if (chunks.find(chunk_id) == chunks.end()) {
                chunks[chunk_id] = parse_chunk(j_chunk);
            }
            layers[layer_id]->chunks_in_.push_back(chunks[chunk_id]);
        }
        for (const json& j_chunk: j_layer["chunks_out"]) {
            string chunk_id = j_chunk["chunk_id"].get<string>();
            if (chunks.find(chunk_id) == chunks.end()) {
                chunks[chunk_id] = parse_chunk(j_chunk);
            }
            layers[layer_id]->chunks_out_.push_back(chunks[chunk_id]);
        }
    }

    net->inputs_.clear();
    net->key_chunks_.clear();
    for (const json& j_chunk: j_net["inputs"]) {
        string chunk_id = j_chunk.get<string>();
        net->inputs_.push_back(chunks[chunk_id]);
    }
    for (const auto& j_key_chunk: j_net["key_chunks"].items()) {
        string key = j_key_chunk.key();
        net->key_chunks_[key] = chunks[j_key_chunk.value()];
    }

    net->all_layers_.clear();
    net->net_graph_.clear();
    net->net_sequences_.clear();
    for (const json& j_layer: j_net["layers"]) {
        string layer_id = j_layer["layer_id"].get<string>();
        net->all_layers_.insert(layers[layer_id]);
        net->net_sequences_.push_back(layers[layer_id]);
        for (const json& j_to: j_layer["to_layers"]) {
            string to_layer_id = j_to.get<string>();
            net->net_graph_[layers[layer_id]].push_back(layers[to_layer_id]);
        }
    }
}

} // namespace micronet
