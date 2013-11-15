#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "adaboost.hpp"

using namespace std;

// update threshold and polarity for classifier in-place.
// the first `num_positive` samples are positive, the rest are negative.
void compute_threshold(weak_classifier &classifier,
                       // feature values per sample for this classifier
                       vector<int> &feature_values,
                       vector<double> &weights,
                       unsigned num_positive) {

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
  vector<double> errors(feature_values.size());
  vector<int> polarities(errors.size());
  double splus = 0;
  double sminus = 0;
  for (int i = indexes.size() - 1; i >= 0; i--) {
    double e1 = splus + tminus - sminus;
    double e2 = sminus + tplus - splus;
    errors[i] = min(e1, e2);
    // polarity
    if (e1 > e2) {
      // lower than threshold --> 1 (means right)
      polarities[i] = 1;
    } else {
      // higher than or equal to threshold --> -1 (means left)
      polarities[i] = -1;
    }
    // update S+ and S-
    double current_weight = weights[indexes[i]];
    if (indexes[i] < num_positive) {
      // positive (add to splus)
      splus = splus + current_weight;
    } else {
      // negative (add to sminus)
      sminus = sminus + current_weight;
    }
  }

  // find the minimum error
  int min_error_index = 0;
  for (int i = 1; i < errors.size(); i++) {
    if (errors[i] < errors[min_error_index]) {
      min_error_index = i;
    }
  }
  classifier.threshold = feature_values[indexes[min_error_index]];
  classifier.polarity  = polarities[min_error_index];
}
