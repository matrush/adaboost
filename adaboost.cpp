#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <queue>
#include <fstream>
#include <map>
#include "adaboost.hpp"


using namespace std;


int main(int argc, char **argv) {
  if (argc != 2) {
    cout << "Usage: ./adaboost <path/to/feature_values>" << endl;
    return 0;
  }

  vector<vector<int> > feature_values = load_2d_array<int>(argv[1]);
  char classifier_filename[22];
  sprintf(classifier_filename, "data/classifier%d.dat", img_size);
  vector<weak_classifier> classifiers = load_array<weak_classifier>(classifier_filename);

  char indexes_filename[22];
  sprintf(indexes_filename, "data/top_index.dat");
  vector<int> indexes = load_array<int>(indexes_filename);

  vector<vector<int> > faces = load_2d_array<int>("data/newface16.dat");
  vector<vector<int> > nonfaces = load_2d_array<int>("data/nonface16.dat");
  vector<vector<int> > samples;

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

  // strong_classifier
  strong_classifier strong(num_iteration);
  printf("%d %d\n", num_faces, num_nonfaces);
  printf("%d %d\n", num_iteration, num_classifier);
  map<pair<int, int>, int> mp;
  for (int t = 0; t < num_iteration; t++) {
    // get errors
    int h_t = -1;
    for (int i = 0; i < num_classifier; i++) {
      if (used[i]) continue;
      errors[i] = compute_threshold(classifiers[indexes[i]],
                                    feature_values[indexes[i]],
                                    weights,
                                    sorted[indexes[i]],
                                    num_faces);
      if (h_t == -1 || errors[h_t] > errors[i]) {
        h_t = i;
      }
    }
    used[h_t] = 1;


    double alpha_t = sgn(errors[h_t]) == 0 ? 0 : 0.5 * log((1.0 - errors[h_t]) / errors[h_t]);
    double z_t = 0;
    for (int i = 0; i < num_samples; i++) {
      int y = (i < num_faces) ? 1 : -1;
      int h = classifiers[indexes[h_t]].h(feature_values[indexes[h_t]][i]);
      weights[i] *= exp(-y * h * alpha_t);
      z_t += weights[i];
    }
    for (int i = 0; i < num_samples; i++) {
      weights[i] /= z_t;
    }
    strong.weak[t] = classifiers[indexes[h_t]];
    strong.weak[t].weight = alpha_t;
    //printf("%.6lf\n", alpha_t);
    printf("%d %.6lf %.6lf\n", indexes[h_t], errors[h_t], alpha_t);
    printf("%u %u %u %u %u %d %d\n", classifiers[indexes[h_t]].x,
                                     classifiers[indexes[h_t]].y,
                                     classifiers[indexes[h_t]].x_size,
                                     classifiers[indexes[h_t]].y_size,
                                     classifiers[indexes[h_t]].id,
                                     classifiers[indexes[h_t]].threshold,
                                     classifiers[indexes[h_t]].polarity);
    if (t == 0 || t == 10 || t == 50 || t == 100 || t == 150 || t == 200) {
      priority_queue<double> q;
      for (int i = 0; i < num_classifier; i++) {
        errors[i] = compute_threshold(classifiers[indexes[i]],
                                      feature_values[indexes[i]],
                                      weights,
                                      sorted[indexes[i]],
                                      num_faces);
        q.push(-errors[i]);
      }
      vector<double> v;
      for (int i = 0; i < 1000; i++) {
        v.push_back(q.top());
        q.pop();
      }
      char filename[50];
      sprintf(filename, "data/top1000_error_at_%d.txt", t);
      FILE *top1000 = fopen(filename, "w");
      for (int i = 0; i < 1000; i++) {
        fprintf(top1000, "%.6lf\n", -v[i]);
      }
      fclose(top1000);
      if (t == 0) continue;

      samples.clear();
      //samples.insert(samples.end(), faces.begin(), faces.begin() + num_faces);
      //samples.insert(samples.end(), nonfaces.begin(), nonfaces.begin() + num_nonfaces);
      samples.insert(samples.end(), faces.begin(), faces.end());
      samples.insert(samples.end(), nonfaces.begin(), nonfaces.end());
      
      sprintf(filename, "data/fx_at_%d.txt", t);
      FILE *Fx = fopen(filename, "w");
      v.clear();
      for (int j = 0; j < samples.size(); j++) {
        double fx = 0;
        for (int i = 0; i < t; i++) {
          fx += strong.weak[i].weight * strong.weak[i].h(compute_feature(samples[j], strong.weak[i]));
        }
        v.push_back(fx);
        fprintf(Fx, "%.6lf\n", fx);
      }
      fclose(Fx);
    }

  }

  // reduce sample size
  //faces.erase(faces.begin(), faces.begin() + num_faces);
  //nonfaces.erase(nonfaces.begin(), nonfaces.begin() + num_nonfaces);
  //faces.resize(num_faces);
  //nonfaces.resize(num_nonfaces);

  // join samples
  samples.clear();
  samples.insert(samples.end(), faces.begin(), faces.begin() + num_faces);
  samples.insert(samples.end(), nonfaces.begin(), nonfaces.begin() + num_nonfaces);
  
  int face_right = 0, face_wrong = 0, nonface_right = 0, nonface_wrong = 0;
  for (int j = 0; j < samples.size(); j++) {
    int y = (j < num_faces) ? 1 : -1;
    int h = strong.H(samples[j]);
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
    int h = strong.H(samples[j]);
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
  save_array<weak_classifier>(strong.weak, "data/strong_classifier.dat");  
}

