#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "adaboost.hpp"

using namespace std;

int get_block_id(int x) {
  if (x < range_left_end) return 0;
  if (x >= range_right_end) return num_blocks - 1;
  return (x - range_left_end) / range_length;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    cout << "Usage: ./realboost <path/to/feature_values>" << endl;
    return 0;
  }

  vector<vector<int> > feature_values = load_2d_array<int>(argv[1]);
  char classifier_filename[22];
  sprintf(classifier_filename, "data/classifier%d.dat", img_size);
  vector<weak_classifier> classifiers = load_array<weak_classifier>(classifier_filename);

  char indexes_filename[22];
  sprintf(indexes_filename, "data/top_index.dat");
  vector<int> indexes = load_array<int>(indexes_filename);
  
  // uniform weights
  vector<double> weights(num_samples, 1.0 / num_samples);

  // errors and used
  vector<double> errors(num_classifier, 0);
  vector<char> used(num_classifier, 0);
  
  // sorted vector
  vector<vector<int> > sorted(num_classifier);
  
  for (int i = 0; i < num_classifier; i++) {
      sorted[indexes[i]].resize(num_samples);
      for (int j = 0; j < num_samples; j++) {
        sorted[indexes[i]][j] = j;
      }
      sort(sorted[indexes[i]].begin(), sorted[indexes[i]].end(), sort_proxy<int>(feature_values[indexes[i]]));
  }
  // real_classifier
  real_classifier real(num_iteration, num_blocks);
  printf("%d %d\n", num_faces, num_nonfaces);
  printf("%d %d %d\n", num_iteration, num_classifier, num_blocks);
  for (int t = 0; t < num_iteration; t++) {
    // get errors
    int h_t = -1;
    for (int i = 0; i < num_classifier; i++) {
      if (used[i]) continue;
      errors[i] = compute_error_real(classifiers[indexes[i]],
                                     feature_values[indexes[i]],
                                     weights,
                                     classifiers[indexes[i]].ht,
                                     num_faces);
      if (h_t == -1 || errors[h_t] > errors[i]) {
        h_t = i;
      }
    }
    used[h_t] = 1;
    double z_t = 0;
    for (int i = 0; i < num_samples; i++) {
      int y = (i < num_faces) ? 1 : -1;
      int id = get_block_id(feature_values[indexes[h_t]][i]);
      double h = classifiers[indexes[h_t]].ht[id];
      weights[i] *= exp(-y * h);
      z_t += weights[i];
    }
    for (int i = 0; i < num_samples; i++) {
      weights[i] /= z_t;
    }
    //printf("%.6lf\n", alpha_t);
    printf("%d %.6lf\n", indexes[h_t], errors[h_t]);
    printf("%u %u %u %u %u %d %d\n", classifiers[indexes[h_t]].x,
                                     classifiers[indexes[h_t]].y,
                                     classifiers[indexes[h_t]].x_size,
                                     classifiers[indexes[h_t]].y_size,
                                     classifiers[indexes[h_t]].id,
                                     classifiers[indexes[h_t]].threshold,
                                     classifiers[indexes[h_t]].polarity);
    real.weak[t] = classifiers[indexes[h_t]];
  }

  vector<vector<int> > faces = load_2d_array<int>("data/newface16.dat");
  vector<vector<int> > nonfaces = load_2d_array<int>("data/nonface16.dat");
  // reduce sample size
  //faces.erase(faces.begin(), faces.begin() + num_faces);
  //nonfaces.erase(nonfaces.begin(), nonfaces.begin() + num_nonfaces);
  //faces.resize(num_faces);
  //nonfaces.resize(num_nonfaces);

  // join samples
  vector<vector<int> > samples;
  samples.insert(samples.end(), faces.begin(), faces.begin() + num_faces);
  samples.insert(samples.end(), nonfaces.begin(), nonfaces.begin() + num_nonfaces);
  
  int face_right = 0, face_wrong = 0, nonface_right = 0, nonface_wrong = 0;
  for (int j = 0; j < samples.size(); j++) {
    int y = (j < num_faces) ? 1 : -1;
    int h = real.H(samples[j]);
    if (y == 1) {
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
  printf("training face: right/wrong = %d/%d\n", face_right, face_wrong);
  printf("training nonface: right/wrong = %d/%d\n", nonface_right, nonface_wrong);
  samples.clear();
  samples.insert(samples.end(), faces.begin() + num_faces, faces.end());
  samples.insert(samples.end(), nonfaces.begin() + num_nonfaces, nonfaces.end());
  
  face_right = 0, face_wrong = 0, nonface_right = 0, nonface_wrong = 0;
  for (int j = 0; j < samples.size(); j++) {
    int y = (j < faces.size() - num_faces) ? 1 : -1;
    int h = real.H(samples[j]);
    if (y == 1) {
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
  printf("testing face: right/wrong = %d/%d\n", face_right, face_wrong);
  printf("testing nonface: right/wrong = %d/%d\n", nonface_right, nonface_wrong);
}

