#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "adaboost.hpp"

using namespace std;

void usage() {
  cout << "Usage: ./topk <K>" << endl;
  exit(0);
}

int main(int argc, char **argv) {

  int k = -1;

  if (argc != 2) {
    usage();
  }

  sscanf(argv[1], "%d", &k);

  if (k <= 0) {
    usage();
  }

  vector<weak_classifier> top2000 = load_array<weak_classifier>("data/top2000.dat");
  FILE *f = fopen("data/topk.txt", "w");
  for (int i = 0; i < k; i++) {
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
