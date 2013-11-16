#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "adaboost.hpp"

using namespace std;

int main(int argc, char **argv) {

  if (argc != 2) {
    cout << "Usage: ./find_errors <path/to/feature_values>" << endl;
    return 0;
  }

  vector<vector<int> > feature_values = load_2d_array<int>(argv[1]);
  char classifier_filename[22];
  sprintf(classifier_filename, "data/classifier%d.dat", img_size);
  vector<weak_classifier> classifiers = load_array<weak_classifier>(classifier_filename);

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
    fprintf(error_f, "%d %lf\n", indexes[i], errors[indexes[i]]);
  }
  fclose(error_f);

  save_array<weak_classifier>(top2000, "data/top2000.dat");
  save_array<int>(indexes, "data/top_index.dat");

  return 0;
}
