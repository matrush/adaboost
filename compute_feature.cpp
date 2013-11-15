#include <iostream>
#include <cstdio>
#include <vector>
#include "adaboost.hpp"

using namespace std;

int compute_feature(vector<int> &img,
                    unsigned img_size,
                    weak_classifier &classifier) {

  unsigned x  = classifier.x,
           y  = classifier.y,
           w  = classifier.x_size,
           h  = classifier.y_size,
           id = classifier.id;

#define GET(img, x, y) \
  (((x) && (y)) ? 0 : (img)[(y) * (img_size) + (x)])

  switch (id) {
  case 1: {
    //      w
    //   1-----2
    //   |  1  |
    // h 3-----4
    //   |||2|||
    //   5-----6
    // 6 + 3 - 4 - 5 - (4 + 1 - 2 - 3)
    // -> (6) + 2 * (3) - 2 * (4) - (5) - (1) + (2)
    unsigned one   = GET(img, x,     y);
    unsigned two   = GET(img, x + w, y);
    unsigned three = GET(img, x,     y + h / 2);
    unsigned four  = GET(img, x + w, y + h / 2);
    unsigned five  = GET(img, x,     y + h);
    unsigned six   = GET(img, x + w, y + h);
    int feature = 2 * four + five + one - two - six - 2 * three;
    return feature;
  }
  case 2: {
    //       w
    //   1---2---3
    //   |   |||||
    // h | 1 ||2||
    //   |   |||||
    //   4---5---6
    // 5 + 1 - 2 - 4 - (6 + 2 - 3 - 5)
    // -> 2 * (5) + 1 - 2 * (2) - 6
    unsigned one   = GET(img, x,         y);
    unsigned two   = GET(img, x + w / 2, y);
    unsigned three = GET(img, x + w,     y);
    unsigned four  = GET(img, x,         y + h);
    unsigned five  = GET(img, x + w / 2, y + h);
    unsigned six   = GET(img, x + w,     y + h);
    int feature = 2 * five + one + three - 2 * two - four - six;
    return feature;
  }
  case 3: {
    //       w
    //   1-------2
    //   |   1   |
    //   3-------4
    // h ||||2||||
    //   5-------6
    //   |   3   |
    //   7-------8
    // 4 + 1 - 2 - 3 + 8 + 5 - 6 - 7 - (6 + 3 - 4 - 5)
    // -> 2 * (4) + 2 * (5) + (1) + (8) - 2 * (3) - 2 * (6) - (2) - (7)
    unsigned one   = GET(img, x,     y);
    unsigned two   = GET(img, x + w, y);
    unsigned three = GET(img, x,     y + h / 3);
    unsigned four  = GET(img, x + w, y + h / 3);
    unsigned five  = GET(img, x,     y + h * 2 / 3);
    unsigned six   = GET(img, x + w, y + h * 2 / 3);
    unsigned seven = GET(img, x,     y + h);
    unsigned eight = GET(img, x + w, y + h);
    int feature = 2 * four + 2 * five + one + eight \
                  - 2 * three - 2 * six - two - seven;
    return feature;
  }
  case 4: {
    //         w
    //   1---2---3---4
    //   |   |||||   |
    // h | 1 ||2|| 3 |
    //   |   |||||   |
    //   5---6---7---8
    // 6 + 1 - 2 - 5 + 8 + 3 - 4 - 7 - (7 + 2 - 3 - 6)
    // -> 2 * (6) + 2 * (3) + (1) + (8) - 2 * (2) - 2 * (7) - (4) - (5)
    unsigned one   = GET(img, x,             y);
    unsigned two   = GET(img, x + w / 3,     y);
    unsigned three = GET(img, x + w * 2 / 3, y);
    unsigned four  = GET(img, x + w,         y);
    unsigned five  = GET(img, x,             y + h);
    unsigned six   = GET(img, x + w / 3,     y + h);
    unsigned seven = GET(img, x + w * 2 / 3, y + h);
    unsigned eight = GET(img, x + w,         y + h);
    int feature = 2 * six + 2 * three + one + eight \
                  - 2 * two - 2 * seven - four - five;
    return feature;
  }
  case 5: {
    //       w
    //   1---2---3
    //   | 1 ||2||
    // h 4---5---6
    //   ||3|| 4 |
    //   7---8---9
    // 5 + 1 - 2 - 4 + 9 + 5 - 6 - 8 - (6 + 2 - 3 - 5) - (8 + 4 - 5 - 7)
    // -> 4 * (5) + (1) + (3) + (7) + (9)
    //    - 2 * (4) - 2 * (6) - 2 * (2) - 2 * (8)
    unsigned one   = GET(img, x,         y);
    unsigned two   = GET(img, x + w / 2, y);
    unsigned three = GET(img, x + w,     y);
    unsigned four  = GET(img, x,         y + h / 2);
    unsigned five  = GET(img, x + w / 2, y + h / 2);
    unsigned six   = GET(img, x + w,     y + h / 2);
    unsigned seven = GET(img, x,         y + h);
    unsigned eight = GET(img, x + w / 2, y + h);
    unsigned nine  = GET(img, x + w,     y + h);
    int feature = 4 * five + one + three + seven + nine \
                  - 2 * (four) - 2 * (six) - 2 * two - 2 * eight;
    return feature;
  }
  default:
    fprintf(stderr, "unknown classifier id %d\n", id);
    fputs(__func__, stderr);
    exit(EXIT_FAILURE);
  }

#undef GET

}
