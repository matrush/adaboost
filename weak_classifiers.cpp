#include <iostream>
#include <cstdio>
#include <vector>
#include <sys/stat.h>

using namespace std;

const int haar_rects[][2] = {
  {1, 2},
  {2, 1},
  {1, 3},
  {3, 1},
  {2, 2}
};

struct weak_classifier {
  unsigned x, y, x_size, y_size;
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
            weak_classifier wc = { x, y, width, height };
            classifiers.push_back(wc);
          }
        }
      }
    }
  }
  return classifiers;
}

void save_classifiers(vector<weak_classifier> &data, const char *filename) {
  FILE *f = fopen(filename, "w");
  if (NULL == f) {
    perror(__func__);
    exit(EXIT_FAILURE);
  }
  int size = data.size();
  fwrite(&size, sizeof(int), 1, f);
  for (int i = 0; i < size; i++) {
    fwrite(&data[i], sizeof(weak_classifier), 1, f);
  }
  fclose(f);
}

vector<weak_classifier> load_classifiers(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (NULL == f) {
    perror(__func__);
    exit(EXIT_FAILURE);
  }
  int size;
  fread(&size, sizeof(int), 1, f);
  vector<weak_classifier> data(size);
  for (int i = 0; i < size; i++) {
    fread(&data[i], sizeof(weak_classifier), 1, f);
  }
  fclose(f);
  return data;
}

int main(int argc, char **argv) {
  vector<weak_classifier> classifier16 = gen_weak_classifiers(16);
  vector<weak_classifier> classifier24 = gen_weak_classifiers(24);
  // save data
  mkdir("data", 0755);
  save_classifiers(classifier16, "data/classifier16.dat");
  save_classifiers(classifier24, "data/classifier24.dat");

  return 0;
}
