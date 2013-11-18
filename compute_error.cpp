#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include "adaboost.hpp"

using namespace std;

double compute_error(weak_classifier &classifier,
                     vector<int> &feature_values,
                     vector<double> &weights,
                     int num_positive) {
  double error = 0;
  for (int i = 0; i < feature_values.size(); i++) {
    // actual
    int y = (i < num_positive) ? 1 : -1;
    // classified
    int h = classifier.h(feature_values[i]);
    if (h != y) {
      error += weights[i];
    }
  }
  return error;
  
  // T+: total sum of positive sample weights
  // T-: total sum of negative sample weights
  double tplus = 0;
  double tminus = 0;
  for (int i = 0; i < num_positive; i++) {
    tplus += weights[i];
  }
  for (int i = num_positive; i < weights.size(); i++) {
    tminus += weights[i];
  }
  assert(fabs(tplus + tminus - 1) < 1e-8);
  vector<int> indexes(feature_values.size());
  for (int i = 0; i < indexes.size(); i++) {
    indexes[i] = i;
  }
   // descending order
  // get sorted index of feature_values
  sort(indexes.begin(), indexes.end(), sort_proxy<int>(feature_values));

  // S+: sum of positive sample weights below the threshold
  // S-: sum of negative sample weights below the threshold
  // e(x) = min((S+) + (T-) - (S-), (S-) + (T+) - (S+))
  double splus = 0;
  double sminus = 0;
  double accumulate_sum_plus = 0;
  double accumulate_sum_minus = 0;
  int last_feature_value = -0x7FFFFFFF;
  for (int i = indexes.size() - 1; i >= 0; i--) {
    // update S+ and S-
    double current_weight = weights[indexes[i]];
    if (feature_values[indexes[i]] == last_feature_value) {
      if (indexes[i] < num_positive) {
        accumulate_sum_plus += current_weight;
      } else {
        accumulate_sum_minus += current_weight;
      }
    } else {
      last_feature_value = feature_values[indexes[i]];
      splus += accumulate_sum_plus;
      sminus += accumulate_sum_minus;
      accumulate_sum_plus = accumulate_sum_minus = 0;
      if (indexes[i] < num_positive) {
        accumulate_sum_plus += current_weight;
      } else {
        accumulate_sum_minus += current_weight;
      }
    }
    double e1 = splus + tminus - sminus;
    double e2 = sminus + tplus - splus;
    if (feature_values[indexes[i]] == classifier.threshold) {
      return min(e1, e2);
    }
  }
  return 0.0;
}

double compute_error_real(weak_classifier &classifier,
                     vector<int> &feature_values,
                     vector<double> &weights,
                     vector<double> &h,
                     int num_positive) {
  vector<double> pt(num_blocks), qt(num_blocks);
  for (int i = 0; i < feature_values.size(); i++) {
    // actual
    int y = (i < num_positive) ? 1 : -1;
    int id = get_block_id(feature_values[i]);
    if (y == 1) {
      pt[id] += weights[i];
    } else {
      qt[id] += weights[i];
    }
  }
  double error = 0;
  for (int i = 0; i < num_blocks; i++) {
    h[i] = 0.5 * log(pt[i] / qt[i]);
    error += 2 * sqrt(pt[i] * qt[i]);
  }
  return error;
}
