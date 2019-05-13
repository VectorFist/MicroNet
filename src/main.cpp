/**
 * @file main.cpp
 * @auther yefajie
 * @data 2018/6/21
 **/
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <fstream>
#include <utility>
#include <thread>
#include <chrono>
#include <string.h>
#include <omp.h>

#include "micronet.h"

using namespace std;
using namespace micronet;
using json = nlohmann::json;


pair<vector<data_t>, vector<data_t>> read_mnist_data() {
    pair<vector<data_t>, vector<data_t>> mnist_data;
    data_t train_images, train_labels;
    read_mnist_images("data/mnist/train-images.idx3-ubyte", train_images);
    read_mnist_lables("data/mnist/train-labels.idx1-ubyte", train_labels);
    for (auto& image: train_images) {
        for (float& pix: image) {
            pix /= 255.0;
        }
    }
    mnist_data.first.push_back(std::move(train_images));
    mnist_data.first.push_back(std::move(train_labels));

    data_t valid_images, valid_labels;
    read_mnist_images("data/mnist/t10k-images.idx3-ubyte", valid_images);
    read_mnist_lables("data/mnist/t10k-labels.idx1-ubyte", valid_labels);
    for (auto& image: valid_images) {
        for (float& pix: image) {
            pix /= 255.0;
        }
    }
    mnist_data.second.push_back(std::move(valid_images));
    mnist_data.second.push_back(std::move(valid_labels));

    return mnist_data;
}

data_t get_noise_data(int noise_dim, int noise_num) {
    data_t noise_data;
    for (int i = 0; i < noise_num; ++i) {
        vector<float> noise(noise_dim);
        normal_random_init(noise_dim, noise.data(), 0.0f, 0.1f);
        noise_data.push_back(noise);
    }
    return noise_data;
}

void save_mnist_imgs(map<string, data_t>& train_data, map<string, data_t>& valid_data) {
    auto& train_imgs = train_data["real"];
    auto& valid_imgs = valid_data["real"];
    json j_imgs;
    j_imgs["train"] = {};
    for (int i = 0; i < train_imgs.size(); ++i) {
        j_imgs["train"].push_back(train_imgs[i]);
    }
    j_imgs["valid"] = {};
    for (int i = 0; i < valid_imgs.size(); ++i) {
        j_imgs["valid"].push_back(valid_imgs[i]);
    }
    ofstream ofs("mnist_imgs.json");
    ofs << j_imgs << endl;
    cout << "save mnist imgs done...";
    exit(0);
}

chunk_ptr graph_constructor(const chunk_ptr& img) {
    auto conv1 = Convolution(5, 5, 1, 1, 6, "valid")(img);
    //auto batch1 = BatchNormalization()(conv1);
    auto relu1 = Activation("relu")(conv1);
    auto pool1 = Pooling(2, 2, 2, 2, "valid", "max")(relu1);

    auto conv2 = Convolution(5, 5, 1, 1, 16, "valid")(pool1);
    //auto batch2 = BatchNormalization()(conv2);
    auto relu2 = Activation("relu")(conv2);
    auto pool2 = Pooling(2, 2, 2, 2, "valid", "max")(relu2);

    auto dense1 = Dense(120)(pool2);
    //auto batch3 = BatchNormalization()(dense1);
    auto relu3 = Activation("relu")(dense1);
    auto dense2 = Dense(84)(relu3);
    //auto batch4 = BatchNormalization()(dense2);
    auto relu4 = Activation("relu")(dense2);
    auto output = Dense(10)(relu4);

    return output;
}

chunk_ptr generator_constructor(const chunk_ptr& noise) {
    auto dense1 = Dense(256)(noise);
    auto batch1 = BatchNormalization()(dense1);
    auto relu1 = Activation("leaky_relu")(batch1);
    auto dense2 = Dense(256)(relu1);
    auto batch2 = BatchNormalization()(dense2);
    auto relu2 = Activation("leaky_relu")(batch2);
    auto dense3 = Dense(28*28)(relu2);
    auto output = Activation("sigmoid")(dense3);

    return output;
}

chunk_ptr discriminator_constructor(const chunk_ptr& real) {
    auto dense1 = Dense(256)(real);
    auto relu1 = Activation("leaky_relu")(dense1);
    auto dense2 = Dense(256)(relu1);
    auto relu2 = Activation("leaky_relu")(dense2);
    auto output = Dense(1)(relu2);

    return output;
}

int main() {
    // Train Gans
    /*pair<vector<data_t>, vector<data_t>> mnist_data = read_mnist_data();
    map<string, data_t> train_data = {{"real", std::move(mnist_data.first[0])}};
    map<string, data_t> valid_data = {{"real", std::move(mnist_data.second[0])}};

    GanNet2 net(generator_constructor, 50, discriminator_constructor, {28*28});
    net.print_net();
    net.set_optimizer(make_shared<AdamOptimizer>(0.001, vector<float>{0.5, 0.75}));
    net.fit(train_data, valid_data, 64, 48, 100, true);
    net.save_model("gannet.json");

    /*GanNet2 net;
    net.load_model("gannet.json");
    map<string, data_t> noise_data = {{"noise", std::move(get_noise_data(50, 100))}};
    data_t infer_data = net.inference(noise_data, 100);
    json j_infer_data = infer_data;
    ofstream ofs("gan_imgs/infer_imgs.json");
    ofs << j_infer_data << endl;
    //net.fit(train_data, valid_data, 64, 48, 100, true);*/


    // Train classification net
    pair<vector<data_t>, vector<data_t>> mnist_data = read_mnist_data();
    map<string, data_t> train_data = {{"img", std::move(mnist_data.first[0])}, {"label", std::move(mnist_data.first[1])}};
    map<string, data_t> valid_data = {{"img", std::move(mnist_data.second[0])}, {"label", std::move(mnist_data.second[1])}};

    ClassifyNet net(graph_constructor, {28, 28, 1});
    net.set_optimizer(make_shared<AdamOptimizer>(0.001, vector<float>{0.5, 0.75}));
    net.fit(train_data, valid_data, 32, 12, 100, true);
    net.save_model("net_mnist.json");


    return 0;
}

