#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include "adaboost.hpp"

using namespace std;

void save_2d_array(vector<vector<int> > &data, const char *filename) {
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
    fwrite(reinterpret_cast<char*>(&data[i][0]), sizeof(int), col, f);
  }
  fclose(f);
}

vector<vector<int> > load_2d_array(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (NULL == f) {
    perror(__func__);
    exit(EXIT_FAILURE);
  }
  int row;
  int col;
  fread(&row, sizeof(int), 1, f);
  fread(&col, sizeof(int), 1, f);
  vector<vector<int> > data(row);
  for (int i = 0; i < row; i++) {
    data[i].resize(col);
    fread(reinterpret_cast<char*>(&data[i][0]), sizeof(int), col, f);
  }
  fclose(f);
  return data;
}

void save_array(vector<weak_classifier> &data, const char *filename) {
  FILE *f = fopen(filename, "w");
  if (NULL == f) {
    perror(__func__);
    exit(EXIT_FAILURE);
  }
  int size = data.size();
  fwrite(&size, sizeof(int), 1, f);
  fwrite(reinterpret_cast<char*>(&data[0]), sizeof(weak_classifier), size, f);
  fclose(f);
}

vector<weak_classifier> load_array(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (NULL == f) {
    perror(__func__);
    exit(EXIT_FAILURE);
  }
  int size;
  fread(&size, sizeof(int), 1, f);
  vector<weak_classifier> data(size);
  fread(reinterpret_cast<char*>(&data[0]), sizeof(weak_classifier), size, f);
  fclose(f);
  return data;
}
