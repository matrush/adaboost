#include <vector>

#ifndef _ADABOOST_H_
#define _ADABOOST_H_

struct weak_classifier {
  unsigned x, y, x_size, y_size, id;
};

int compute_feature(std::vector<int> &image,
                    unsigned img_size,
                    weak_classifier &classifier);

///////// 2D array load/save
void save_2d_array(std::vector<std::vector<int> > &data,
                   const char *filename);

std::vector<std::vector<int> > load_2d_array(const char *filename);

//////// 1D array save/load
void save_array(std::vector<weak_classifier> &data,
                const char *filename);

std::vector<weak_classifier> load_array(const char *filename);

#endif /* end of adaboost.hpp */
