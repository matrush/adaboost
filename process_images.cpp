#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <glob.h>
#include <sys/stat.h>
#include "adaboost.hpp"

using namespace std;

inline vector<string> my_glob(const char *pat){
  glob_t glob_result;
  glob(pat, GLOB_TILDE, NULL, &glob_result);
  vector<string> ret;
  for(unsigned int i = 0; i < glob_result.gl_pathc; ++i) {
    ret.push_back(string(glob_result.gl_pathv[i]));
  }
  globfree(&glob_result);
  return ret;
}

// assume that three channels are equal
vector<int> read_bmp(const char *filename) {
  FILE* f = fopen(filename, "r");
  unsigned char info[54];
  fread(info, sizeof(unsigned char), 54, f);

  int width = *(int*)&info[18];
  int height = *(int*)&info[22];

  int size = 3 * width * height;
  unsigned char* data = new unsigned char[size];
  fread(data, sizeof(unsigned char), size, f);
  fclose(f);

  vector<int> container(height * width);
  for (int i = 0; i < size; i += 3) {
    container[((width - 1) - i / width / 3) * width + (i / 3 % width)] = data[i];
    // container.push_back(data[i]);
  }

  delete[] data;
  return container;
}

void test_output_images(vector<int>&  data) {
  int img_size = 16;
  for (int i = 0; i < img_size; i++) {
    for (int j = 0; j < img_size; j++) {
      if (j) putchar(' ');
      printf("%03d", data[i * img_size + j]);
    }
    puts("");
  }
  puts("");
}
// REMEMBER to delete the matrix
// @size: 16 or 24
vector< vector<int> > read_bmp_glob(const char *path) {
  vector<string> files = my_glob(path);
  vector< vector<int> > samples(files.size());
  for (int i = 0; i < files.size(); i++) {
    samples[i] = read_bmp(files[i].c_str());
  }
  /*for (int i = 0; i < 5; i++) {
    test_output_images(samples[i]);
  }*/
  return samples;
}

void integral_images(vector<vector<int> > &samples) {
  int size = (samples[0].size() == 16 * 16) ? 16 : 24;
  for (int i = 0; i < samples.size(); i++) {
    // per row
    for (int j = 0; j < size; j++) {
      // cumulative sum per row (but k is column number)
      for (int k = 1; k < size; k++) {
        samples[i][j * size + k] += samples[i][j * size + k - 1];
      }
    }
    // per column
    for (int j = 0; j < size; j++) {
      // cumulative sum per column (but k is row number)
      for (int k = 1; k < size; k++) {
        samples[i][k * size + j] += samples[i][(k - 1) * size + j];
      }
    }
  }
}

int main(int argc, char **argv) {
  vector<vector<int> > newface16 = read_bmp_glob("samples/newface16/*.bmp");
  vector<vector<int> > newface24 = read_bmp_glob("samples/newface24/*.bmp");
  vector<vector<int> > nonface16 = read_bmp_glob("samples/nonface16/*.bmp");
  vector<vector<int> > nonface24 = read_bmp_glob("samples/nonface24/*.bmp");

  // integral images
  integral_images(newface16);
  integral_images(newface24);
  integral_images(nonface16);
  integral_images(nonface24);

  // write into something
  mkdir("data", 0755);
  save_2d_array<int>(newface16, "data/newface16.dat");
  save_2d_array<int>(newface24, "data/newface24.dat");
  save_2d_array<int>(nonface16, "data/nonface16.dat");
  save_2d_array<int>(nonface24, "data/nonface24.dat");

  return 0;
}
