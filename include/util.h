#ifndef UTIL_H
#define UTIL_H
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <chrono>
#include <random>
#include <map>

#include "nlohmann/json.hpp"
#include "net.h"
#include "math_func.h"

using namespace std;
using json = nlohmann::json;

namespace micronet {

void read_mnist_lables(const string& filename, vector<vector<float>>& labels);
void read_mnist_images(const string& filename, vector<vector<float>>& images);
void read_cifar10_images(const string& dirname, vector<vector<float>>& images, bool train=true);
void read_cifar100_images(const string& dirname, vector<vector<float>>& images);
void read_cifar10_gan_imgs(const string& dirname, vector<vector<float>>& images, int category);

void cal_low_imgs(const vector<vector<float>>& images, vector<vector<float>>& low_images, float factor=0.5);
void cal_L_component_from_RGB(const vector<vector<float>>& rgb_images, vector<vector<float>>& l_images);

void img2col(const float* data_im, int channels,  int height,  int width, int ksize_h,
            int ksize_w, int pad_h, int pad_w, int stride_h, int stride_w, float* data_col);
void col2img(const float* data_col, int channels, int height, int width, int ksize_h,
             int ksize_w, int pad_h, int pad_w, int stride_h, int stride_w, float* data_im);

void normal_random_init(int n, float* x, float mean, float seddev, int seed=-1);
void uniform_random_init(int n, float* x, float lower, float upper, int seed=-1);
void constant_init(int n, float* x, float val);
//std::uniform_real_distribution<float> get_random_uniform_generator(float lower, float upper, int seed=-1);
//std::normal_distribution<float> get_random_normal_generator(float mean, float stddev, int seed=-1);

struct Timer {
    std::chrono::high_resolution_clock::time_point time_start, time_stop;
    Timer() {
        start();
    }
    void start() {
        time_start = std::chrono::high_resolution_clock::now();
    }
    void stop() {
        time_stop = std::chrono::high_resolution_clock::now();
    }
    void resume() {
        time_start = std::chrono::high_resolution_clock::now();
    }
    double elapsed() {
        stop();
        auto span = std::chrono::duration_cast<std::chrono::duration<double>>(time_stop - time_start);
        return span.count();
    }
};

struct UniformGenerator {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution;
    UniformGenerator(float lower, float upper, int seed=-1): distribution(lower, upper) {
        if (seed == -1) {
            seed = std::chrono::system_clock::now().time_since_epoch().count();
        }
        generator = std::default_random_engine(seed);
    }
    float operator()() {
        return distribution(generator);
    }
};

struct NormalGenerator {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution;
    NormalGenerator(float mean, float stddev, int seed=-1): distribution(mean, stddev) {
        if (seed == -1) {
            seed = std::chrono::system_clock::now().time_since_epoch().count();
        }
        generator = std::default_random_engine(seed);
    }
    float operator()() {
        return distribution(generator);
    }
};

void to_json(json& j_net, Net* net);
void from_json(const json& j_net, Net* net);

} // namespace micronet

#endif // UTIL_H
