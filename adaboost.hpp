#ifndef _ADABOOST_H_
#define _ADABOOST_H_

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;

struct weak_classifier {
  unsigned x, y, x_size, y_size, id;
  int threshold, polarity;
};

///////// utility functions
int compute_feature(vector<int> &image,
                    unsigned img_size,
                    weak_classifier &classifier);

void compute_threshold(weak_classifier &classifier,
                       // feature values per sample for this classifier
                       vector<int> &feature_values,
                       vector<double> &weights,
                       unsigned num_positive);

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
