#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "adaboost.hpp"

using namespace std;

int main(int argc, char **argv) {
  if (argc != 2) {
    cout << "Usage: ./adaboost <path/to/feature_values>" << endl;
    return 0;
  }

  vector<vector<int> > feature_values = load_2d_array<int>(argv[1]);
  char classifier_filename[22];
  sprintf(classifier_filename, "data/classifier%u.dat", img_size);
  vector<weak_classifier> classifiers = load_array<weak_classifier>(classifier_filename);

  char indexes_filename[22];
  sprintf(indexes_filename, "data/top_index.dat");
  vector<int> indexes = load_array<int>(indexes_filename);
  /*for (int i = 0; i < indexes.size(); i++) {
    printf("%d\n", indexes[i]);
    printf("%u %u %u %u %u %d %d\n", classifiers[indexes[i]].x,
                                     classifiers[indexes[i]].y,
                                     classifiers[indexes[i]].x_size,
                                     classifiers[indexes[i]].y_size,
                                     classifiers[indexes[i]].id,
                                     classifiers[indexes[i]].threshold,
                                     classifiers[indexes[i]].polarity);
  }*/
  // uniform weights
  vector<double> weights(num_samples, 1.0 / num_samples);

  // errors and used
  vector<double> errors(num_classifier, 0);
  vector<bool> used(num_classifier, false);

  //
  // strong_classifier
  strong_classifier strong(num_iteration);
  printf("%d %d\n", num_faces, num_nonfaces);
  printf("%d %d\n", num_iteration, num_classifier);
  for (int t = 0; t < num_iteration; t++) {
    // get errors
    int h_t = -1;
    for (int i = 0; i < num_classifier; i++) {
      if (used[i]) continue;
      compute_threshold(classifiers[indexes[i]],
                        feature_values[indexes[i]],
                        weights,
                        num_faces);
      errors[i] = compute_error(classifiers[indexes[i]],
                                feature_values[indexes[i]],
                                weights,
                                num_faces);
      if (h_t == -1 || errors[h_t] < errors[i]) {
        h_t = i;
      }
    }
    printf("%d %.6lf\n", indexes[h_t], errors[h_t]);
    printf("%u %u %u %u %u %d %d\n", classifiers[indexes[h_t]].x,
                                     classifiers[indexes[h_t]].y,
                                     classifiers[indexes[h_t]].x_size,
                                     classifiers[indexes[h_t]].y_size,
                                     classifiers[indexes[h_t]].id,
                                     classifiers[indexes[h_t]].threshold,
                                     classifiers[indexes[h_t]].polarity);
    used[h_t] = true;
    double alpha_t = 0.5 * log((1.0 - errors[h_t]) / errors[h_t]);
    for (int i = 0; i < num_samples; i++) {
      int y = (i < num_faces) ? 1 : -1;
      int h = classifiers[indexes[h_t]].h(feature_values[indexes[h_t]][i]);
      weights[i] *= exp(-y * h * alpha_t);
    }
    double z_t = accumulate(weights.begin(), weights.end(), 0.0);
    for (int i = 0; i < num_samples; i++) {
      weights[i] /= z_t;
    }
    strong.weak[t] = classifiers[indexes[h_t]];
    strong.alpha_t[t] = alpha_t;
    //printf("%.6lf\n", alpha_t);
  }

  vector<vector<int> > faces = load_2d_array<int>("data/newface16.dat");
  vector<vector<int> > nonfaces = load_2d_array<int>("data/nonface16.dat");
  // reduce sample size
  //faces.resize(num_faces);
  //nonfaces.resize(num_nonfaces);

  // join samples
  vector<vector<int> > samples;
  samples.insert(samples.end(), faces.begin(), faces.end());
  samples.insert(samples.end(), nonfaces.begin(), nonfaces.end());

  int face_right = 0, face_wrong = 0, nonface_right = 0, nonface_wrong = 0;
  for (int j = 0; j < samples.size(); j++) {
    int y = (j < faces.size()) ? 1 : -1;
    int h = strong.H(samples[j]);
    if (y == 1) {
      //printf("%d %d\n", y, h);
      if (h == 1) {
        face_right++;
      } else {
        face_wrong++;
      }
    } else {
      if (h == -1) {
        nonface_right++;
      } else {
        nonface_wrong++;
      }
    }
  }
  printf("face: right/wrong = %d/%d\n", face_right, face_wrong);
  printf("nonface: right/wrong = %d/%d\n", nonface_right, nonface_wrong);
}

