/**
 * @file optimizer.cpp
 * @auther yefajie
 * @data 2018/6/25
 **/
#include "optimizer.h"

Optimizer::Optimizer(float learning_rate, const vector<int>& decay_steps):
    learning_rate_(learning_rate), decay_steps_(decay_steps) {
}
