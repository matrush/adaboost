#include <iostream>
#include <cstdio>
#include <vector>
#include <sys/stat.h>
#include "adaboost.hpp"

using namespace std;

const int haar_rects[][2] = {
  {1, 2},
  {2, 1},
  {1, 3},
  {3, 1},
  {2, 2}
};

vector<weak_classifier> gen_weak_classifiers(unsigned frame_size) {
  vector<weak_classifier> classifiers;
  int n_rects = sizeof(haar_rects) / sizeof(haar_rects[0]);
  for (int i = 0; i < n_rects; i++) {
    int x_size = haar_rects[i][0];
    int y_size = haar_rects[i][1];
    // for each size
    for (int width = x_size; width <= frame_size; width += x_size) {
      for (int height = y_size; height <= frame_size; height += y_size) {
        for (int x = 0; x <= frame_size - width; x++) {
          for (int y = 0; y <= frame_size - height; y++) {
            weak_classifier wc = { x, y, width, height, i + 1 };
            classifiers.push_back(wc);
          }
        }
      }
    }
  }
  return classifiers;
}

int main(int argc, char **argv) {
  vector<weak_classifier> classifier16 = gen_weak_classifiers(16);
  vector<weak_classifier> classifier24 = gen_weak_classifiers(24);
  // save data
  mkdir("data", 0755);
  save_array<weak_classifier>(classifier16, "data/classifier16.dat");
  save_array<weak_classifier>(classifier24, "data/classifier24.dat");

  return 0;
}
