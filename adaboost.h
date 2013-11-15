#ifndef _ADABOOST_H_
#define _ADABOOST_H_

struct weak_classifier {
  unsigned x, y, x_size, y_size, id;
};

int compute_feature(vector<int> &image,
                    unsigned img_size,
                    weak_classifier &classifier);

#endif /* end of adaboost.h */
