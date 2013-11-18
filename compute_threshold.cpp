#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include "adaboost.hpp"

using namespace std;

// update threshold and polarity for classifier in-place.
// the first `num_positive` samples are positive, the rest are negative.
double compute_threshold(weak_classifier &classifier,
                       // feature values per sample for this classifier
                       vector<int> &feature_values,
                       vector<double> &weights,
                       vector<int> &indexes,
                       int num_positive) {

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
  if (indexes.size() == 0) {
    indexes.resize(feature_values.size());
    for (int i = 0; i < indexes.size(); i++) {
      indexes[i] = i;
    }
    sort(indexes.begin(), indexes.end(), sort_proxy<int>(feature_values));
  }

  // S+: sum of positive sample weights below the threshold
  // S-: sum of negative sample weights below the threshold
  // e(x) = min((S+) + (T-) - (S-), (S-) + (T+) - (S+))
  vector<double> errors(feature_values.size());
  vector<int> polarities(errors.size());
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
      /*if (indexes[i] < num_positive) {
        // positive (add to splus)
        splus += accumulate_sum_plus;
        accumulate_sum_plus = current_weight;
      } else {
        // negative (add to sminus)
        sminus += accumulate_sum_minus;
        accumulate_sum_minus = current_weight;
      }*/
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
    assert(fabs(e1 + e2 - 1) < 1e-8);
    errors[i] = min(e1, e2);
    // polarity
    if (e1 > e2) {
      // lower than threshold --> -1 (means right)
      polarities[i] = -1;
    } else {
      // higher than or equal to threshold --> 1 (means left)
      polarities[i] = 1;
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
  double xx = compute_error(classifier, feature_values, weights, num_positive);
  if (xx > 0.5) {
    printf("haha %d\n", min_error_index);
    printf("%.6lf %.6lf\n", tplus, tminus);
    for (int i = errors.size() - 1; i >= 0; i--) {
      printf("%lf:%d ", errors[i], polarities[i]);
    }
    puts("");
    printf("%.6lf %.6lf\n", errors[min_error_index], xx);
    printf("%d %d\n", classifier.threshold, classifier.polarity);
    double sum = 0.0;
    for (int i = 0; i < feature_values.size(); i++) {
      int y = indexes[i] < num_positive ? 1 : -1;
      int p = classifier.h(feature_values[indexes[i]]);//feature_values[indexes[i]] >= classifier.threshold ? classifier.polarity : -classifier.polarity;
      printf("%d:%d:%d ", feature_values[indexes[i]], y, p);
      if (p != y) {
        sum += weights[indexes[i]];
      }
    }
    printf("hehe %.6lf\n", sum);
    puts("");
    puts("@");
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
      printf("%d:%.6lf:%.6lf sp = %.6lf sm = %.6lf ap = %.6lf am = %.6lf\n", i, e1, e2, splus, sminus, accumulate_sum_plus, accumulate_sum_minus);
    }
    puts("");
    puts("#");

  }
  return errors[min_error_index];
}
