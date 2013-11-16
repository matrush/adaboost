#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "adaboost.hpp"

using namespace std;

int main(int argc, char **argv) {

  if (argc != 2) {
    cout << "Usage: ./precompute_feature_values <path/to/output>" << endl;
    return 0;
  }

  // load samples
  vector<vector<int> > faces = load_2d_array<int>("data/newface16.dat");
  vector<vector<int> > nonfaces = load_2d_array<int>("data/nonface16.dat");
  // reduce sample size
  faces.resize(num_faces);
  nonfaces.resize(num_nonfaces);
  // join samples
  vector<vector<int> > samples;
  samples.insert(samples.end(), faces.begin(), faces.end());
  samples.insert(samples.end(), nonfaces.begin(), nonfaces.end());

  // load weak classifiers
  vector<weak_classifier> classifiers = load_array<weak_classifier>("data/classifier16.dat");
  int num_classifiers = classifiers.size();

  // happy
  // classifiers x samples
  vector<vector<int> > feature_values(num_classifiers);
  for (int i = 0; i < num_classifiers; i++) {
    feature_values[i].resize(num_samples);
    for (int j = 0; j < num_samples; j++) {
      feature_values[i][j] = compute_feature(samples[j],
                                             classifiers[i]);
    }
  }

  // save data
  save_2d_array<int>(feature_values, argv[1]);
  return 0;
}
