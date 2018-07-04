#ifndef UTIL_H
#define UTIL_H
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

using namespace std;

void read_mnist_lables(const string& filename, vector<float>& labels);
void read_mnist_images(const string& filename, vector<vector<float>>& images);

#endif // UTIL_H
