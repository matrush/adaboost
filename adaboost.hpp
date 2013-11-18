#ifndef _ADABOOST_H_
#define _ADABOOST_H_

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;

const double eps = 1e-6;

inline int sgn(double x) {
  return x < -eps ? -1 : x > eps;
}

const int num_faces = 10000;
const int num_nonfaces = 10000;
const int num_samples = num_faces + num_nonfaces;
const int img_size = 16;
const int num_iteration = 200;
const int num_classifier = 32384;

struct weak_classifier;

///////// utility functions
int compute_feature(vector<int> &image,
                    weak_classifier &classifier);

double compute_threshold(weak_classifier &classifier,
                       // feature values per sample for this classifier
                       vector<int> &feature_values,
                       vector<double> &weights,
                       vector<int> &indexes,
                       int num_positive);

double compute_error(weak_classifier &classifier,
                     vector<int> &feature_values,
                     vector<double> &weights,
                     int num_positive);

double compute_error_real(weak_classifier &classifier,
                     vector<int> &feature_values,
                     vector<double> &weights,
                     int num_positive);

///////// classifiers
struct weak_classifier {
  int x, y, x_size, y_size, id;
  int threshold, polarity;
  int h(int x) {
    if (polarity == 1) {
      return x >= threshold ? 1 : -1;
    } else {
      return x < threshold ? 1 : -1;
    }
  }
};

struct strong_classifier {
  int T;
  vector<double> alpha_t;
  vector<weak_classifier> weak;
  strong_classifier(int t):T(t) {
    alpha_t.resize(t);
    weak.resize(t);
  }
  int H(int x) {
    double fx = 0;
    for (int i = 0; i < T; i++) {
      fx += alpha_t[i] * weak[i].h(x);
    }
    return sgn(fx) >= 0 ? 1 : -1;
  }
  int H(vector<int> &sample) {
    double fx = 0;
    for (int i = 0; i < T; i++) {
      fx += alpha_t[i] * weak[i].h(compute_feature(sample, weak[i]));
    }
    return sgn(fx) >= 0 ? 1 : -1;
  }
};

///////// sort proxy
template <typename T>
struct sort_proxy {
  sort_proxy(vector<T> &vec) : values(vec) {}
  bool operator() (const int& a, const int& b) const {
    // descending order
    return values[a] > values[b];
  }
  vector<T>& values;
};

///////// 2D array load/save
template<typename T>
void save_2d_array(vector<vector<T> > &data, const char *filename) {
  FILE *f = fopen(filename, "w");
  if (NULL == f) {
    perror(__func__);
    exit(EXIT_FAILURE);
  }
  int row = data.size();
  int col = data[0].size();
  fwrite(&row, sizeof(int), 1, f);
  fwrite(&col, sizeof(int), 1, f);
  for (int i = 0; i < row; i++) {
    fwrite(reinterpret_cast<char*>(&data[i][0]), sizeof(T), col, f);
  }
  fclose(f);
}

template<typename T>
vector<vector<T> > load_2d_array(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (NULL == f) {
    perror(__func__);
    exit(EXIT_FAILURE);
  }
  int row;
  int col;
  fread(&row, sizeof(int), 1, f);
  fread(&col, sizeof(int), 1, f);
  vector<vector<T> > data(row);
  for (int i = 0; i < row; i++) {
    data[i].resize(col);
    fread(reinterpret_cast<char*>(&data[i][0]), sizeof(T), col, f);
  }
  fclose(f);
  return data;
}

//////// 1D array save/load
template<typename T>
void save_array(vector<T> &data, const char *filename) {
  FILE *f = fopen(filename, "w");
  if (NULL == f) {
    perror(__func__);
    exit(EXIT_FAILURE);
  }
  int size = data.size();
  fwrite(&size, sizeof(int), 1, f);
  fwrite(reinterpret_cast<char*>(&data[0]), sizeof(T), size, f);
  fclose(f);
}

template<typename T>
vector<T> load_array(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (NULL == f) {
    perror(__func__);
    exit(EXIT_FAILURE);
  }
  int size;
  fread(&size, sizeof(int), 1, f);
  vector<T> data(size);
  fread(reinterpret_cast<char*>(&data[0]), sizeof(T), size, f);
  fclose(f);
  return data;
}

#endif /* end of adaboost.hpp */
