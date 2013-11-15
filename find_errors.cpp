#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "adaboost.hpp"

using namespace std;

const int num_faces = 5000;
const int num_nonfaces = 10000;
const unsigned img_size = 16;

double compute_error(weak_classifier &classifier,
                     vector<int> &feature_values,
                     vector<double> &weights,
                     unsigned num_positive) {
  int p = classifier.polarity, t = classifier.threshold;
  double error = 0;
  for (int i = 0; i < feature_values.size(); i++) {
    // actual
    int y = (i < num_positive) ? 1 : -1;
    // classified
    int h = (p * t > p * feature_values[i]) ? 1 : -1;
    error += weights[i] * abs(h - y) / 2;
  }
  return error;
}

int main(int argc, char **argv) {

  if (argc != 2) {
    cout << "Usage: ./find_errors <path/to/feature_values>" << endl;
    return 0;
  }

  vector<vector<int> > feature_values = load_2d_array<int>(argv[1]);
  vector<weak_classifier> classifiers = load_array<weak_classifier>("data/classifier16.dat");

  // uniform weights
  vector<double> weights(feature_values[0].size(), 1.0 / feature_values[0].size());

  // update threshold and polarity for all classifiers
  for (int i = 0; i < classifiers.size(); i++) {
    compute_threshold(classifiers[i],
                      feature_values[i],
                      weights,
                      num_faces);
  }

  // compute error for each classifiers
  vector<double> errors(classifiers.size());
  for (int i = 0; i < classifiers.size(); i++) {
    errors[i] = compute_error(classifiers[i],
                              feature_values[i],
                              weights,
                              num_faces);
  }

  // sort by error
  vector<int> indexes(feature_values.size());
  for (int i = 0; i < indexes.size(); i++) {
    indexes[i] = i;
  }
  // descending order
  // get sorted index of errors
  sort(indexes.begin(), indexes.end(), sort_proxy<double>(errors));

  // top 2000 classifiers
  reverse(indexes.begin(), indexes.end());

  vector<weak_classifier> top2000(2000);
  for (int i = 0; i < 2000; i++) {
    top2000[i] = classifiers[indexes[i]];
  }

  // save errors
  FILE *error_f = fopen("data/top2000-errors.txt", "w");
  for (int i = 0; i < 2000; i++) {
    fprintf(error_f, "%lf\n", errors[indexes[i]]);
  }
  fclose(error_f);

  save_array<weak_classifier>(top2000, "data/top2000.dat");

  return 0;
}
