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

#include "chunk.h"
#include "layer.h"
#include "convolution.h"
#include "bias.h"
#include "softmax.h"
#include "softmaxloss.h"
#include "dense.h"
#include "relu.h"
#include "argmax.h"
#include "pooling.h"
#include "accuracy.h"
#include "data.h"
#include "net.h"
#include "util.h"
#include "sgdoptimizer.h"
#include "adagradoptimizer.h"
#include "rmsproboptimizer.h"
#include "adamoptimizer.h"

using namespace std;


void print(const Chunk& chunk, const string& name) {
    const float* data = chunk.const_data();
    const float* diff = chunk.const_diff();
    cout << name << " shape" << endl;
    cout << chunk.str_shape() << endl;
    if (name == "loss" || name == "accuracy") {
        cout << "data" << endl;
        for (int i = 0; i < chunk.count(); ++i) {
            cout << data[i] << " ";
        }
        cout << endl << "diff" << endl;
        for (int i = 0; i < chunk.count(); ++i) {
            cout << diff[i] << " ";
        }
    }
    cout << endl << endl;
}

void print_loss_acc(int iter, const Chunk& loss, const Chunk& acc, double time_per_iter) {
    const float* loss_data = loss.const_data();
    const float* acc_data = acc.const_data();
    cout << "iter: " << setw(5) << iter << "\tloss: " << fixed <<setprecision(5) <<
        loss_data[0] << "\tacc: " << fixed << setprecision(4) << acc_data[0]  <<
        "\ttime_per_iter: " << fixed << setprecision(4) << (time_per_iter * 1000) << " ms" << endl;
}

void write_to_file(map<string, vector<float>>& acc_map, map<string, vector<float>>& loss_map,
                   string optimizer) {
    ofstream acc_file(optimizer + "_acc_file.txt", std::ios::out);
    ofstream loss_file(optimizer + "_loss_file.txt", std::ios::out);
    acc_file << "train";
    for (float acc: acc_map["train"]) {
        acc_file << " " << acc;
    }
    acc_file << endl;
    acc_file << "test";
    for (float acc: acc_map["test"]) {
        acc_file << " " << acc;
    }

    loss_file << "train";
    for (float loss: loss_map["train"]) {
        loss_file << " " << loss;
    }
    loss_file << endl;
    loss_file << "test";
    for (float loss: loss_map["test"]) {
        loss_file << " " << loss;
    }
    acc_file.close();
    loss_file.close();
}

int main()
{
    Net net("LeNet5");

    shared_ptr<Layer> data_train(new Data("data", 32, "data_train_layer", "train"));
    data_train->set_chunks(vector<string>{"null"}, vector<string>{"img", "label"});
    net.add_layer(data_train);

    shared_ptr<Layer> data_test(new Data("data", 100, "data_test_layer", "test"));
    data_test->set_chunks(vector<string>{"null"}, vector<string>{"img", "label"});
    net.add_layer(data_test);

    shared_ptr<Layer> conv1(new Convolution(5, 5, 2, 2, 1, 1, 1, 6, "conv1_layer"));
    conv1->set_chunks(vector<string>{"img"}, vector<string>{"conv1"});
    net.add_layer(conv1);

    shared_ptr<Layer> relu1(new ReLU("relu1_layer"));
    relu1->set_chunks(vector<string>{"conv1"}, vector<string>{"relu1"});
    net.add_layer(relu1);

    shared_ptr<Layer> pool1(new Pooling(2, 2, 0, 0, 2, 2, "pool1_layer", "avg"));
    pool1->set_chunks(vector<string>{"relu1"}, vector<string>{"pool1"});
    net.add_layer(pool1);

    shared_ptr<Layer> conv2(new Convolution(5, 5, 0, 0, 1, 1, 6, 16, "conv2_layer"));
    conv2->set_chunks(vector<string>{"pool1"}, vector<string>{"conv2"});
    net.add_layer(conv2);

    shared_ptr<Layer> relu2(new ReLU("relu2_layer"));
    relu2->set_chunks(vector<string>{"conv2"}, vector<string>{"relu2"});
    net.add_layer(relu2);

    shared_ptr<Layer> pool2(new Pooling(2, 2, 0, 0, 2, 2, "pool2_layer", "avg"));
    pool2->set_chunks(vector<string>{"relu2"}, vector<string>{"pool2"});
    net.add_layer(pool2);

    shared_ptr<Layer> dense1(new Dense(16*5*5, 120, "dense1_layer"));
    dense1->set_chunks(vector<string>{"pool2"}, vector<string>{"dense1"});
    net.add_layer(dense1);

    shared_ptr<Layer> relu3(new ReLU("relu3_layer"));
    relu3->set_chunks(vector<string>{"dense1"}, vector<string>{"relu3"});
    net.add_layer(relu3);

    shared_ptr<Layer> dense2(new Dense(120, 84, "dense2_layer"));
    dense2->set_chunks(vector<string>{"relu3"}, vector<string>{"dense2"});
    net.add_layer(dense2);

    shared_ptr<Layer> relu4(new ReLU("relu4_layer"));
    relu4->set_chunks(vector<string>{"dense2"}, vector<string>{"relu4"});
    net.add_layer(relu4);

    shared_ptr<Layer> output(new Dense(84, 10, "output_layer"));
    output->set_chunks(vector<string>{"relu4"}, vector<string>{"output"});
    net.add_layer(output);

    shared_ptr<Layer> softmaxloss(new SoftmaxLoss("softmaxloss_layer"));
    softmaxloss->set_chunks(vector<string>{"output", "label"}, vector<string>{"loss", "prob"});
    net.add_layer(softmaxloss);

    shared_ptr<Layer> accuracy(new Accuracy("accuracy_layer"));
    accuracy->set_chunks(vector<string>{"prob", "label"}, vector<string>{"accuracy"});
    net.add_layer(accuracy);

    net.set_data_layers(vector<string>{"data_train_layer", "data_test_layer"});
    net.initialize();
    net.set_default_data_layer("data_train_layer");

    shared_ptr<Optimizer> optimizer(new AdamOptimizer(0.001, vector<int>{6000, 8000}));
    net.set_optimizer(optimizer);

    map<string, vector<float>> acc_map;
    map<string, vector<float>> loss_map;

    int test_interval = 100;
    for (int i = 0; i < 10000; ++i) {
        auto time_point1 = std::chrono::high_resolution_clock::now();

        net.forward();
        net.backward();
        net.update();

        auto time_point2 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(time_point2 - time_point1);

        if (i % 10 == 0) {
            print_loss_acc(i, net.all_chunks_["loss"], net.all_chunks_["accuracy"], time_span.count());
            acc_map["train"].push_back(net.all_chunks_["accuracy"].const_data()[0]);
            loss_map["train"].push_back(net.all_chunks_["loss"].const_data()[0]);
        }
        if (i % test_interval == 0) {
            net.set_default_data_layer("data_test_layer");
            float test_acc = 0;
            float test_loss = 0;
            for (int j = 0; j < 100; ++j) {
                net.forward();
                test_acc += net.all_chunks_["accuracy"].const_data()[0];
                test_loss += net.all_chunks_["loss"].const_data()[0];
            }
            test_acc /= 100;
            test_loss /= 100;
            acc_map["test"].push_back(test_acc);
            loss_map["test"].push_back(test_loss);
            cout << "-------------------------- test acc: " << test_acc << "---------------------------" << endl;
            net.set_default_data_layer("data_train_layer");
        }
    }
    write_to_file(acc_map, loss_map, "adam");

    return 0;
}

