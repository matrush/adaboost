#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "adaboost.hpp"

using namespace std;

int main(int argc, char **argv) {
  vector<weak_classifier> top2000 = load_array<weak_classifier>("data/top2000.dat");
  FILE *f = fopen("data/top10.txt", "w");
  for (int i = 0; i < 10; i++) {
    fprintf(f, "%u %u %u %u %u %d %d\n", top2000[i].x,
                                         top2000[i].y,
                                         top2000[i].x_size,
                                         top2000[i].y_size,
                                         top2000[i].id,
                                         top2000[i].threshold,
                                         top2000[i].polarity);
  }
  fclose(f);
  return 0;
}
