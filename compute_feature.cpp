#include <iostream>
#include <cstdio>
#include <vector>
#include "adaboost.hpp"

using namespace std;

int compute_feature(vector<int> &img,
                    weak_classifier &classifier) {
       int x  = classifier.x - 1,
           y  = classifier.y - 1,
           w  = classifier.x_size,
           h  = classifier.y_size,
           id = classifier.id;

#define GET(img, x, y) \
  (((x) == -1 || (y) == -1) ? 0 : (img)[(y) * (img_size) + (x)])

  switch (id) {
  case 1: {
    //      w
    //   1-----2
    //   |||1|||
    // h 3-----4
    //   |  2  |
    //   5-----6
    // 6 + 3 - 4 - 5 - (4 + 1 - 2 - 3)
    // -> (6) + 2 * (3) - 2 * (4) - (5) - (1) + (2)
    int one   = GET(img, x,     y);
    int two   = GET(img, x + w, y);
    int three = GET(img, x,     y + h / 2);
    int four  = GET(img, x + w, y + h / 2);
    int five  = GET(img, x,     y + h);
    int six   = GET(img, x + w, y + h);
    int feature = six + 2 * three - 2 * four - five - one + two;
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
    // -> 2 * (5) + 1 - 2 * (2) - 6 + 3 - 4
    int one   = GET(img, x,         y);
    int two   = GET(img, x + w / 2, y);
    int three = GET(img, x + w,     y);
    int four  = GET(img, x,         y + h);
    int five  = GET(img, x + w / 2, y + h);
    int six   = GET(img, x + w,     y + h);
    int feature = 2 * five + one - 2 * two - six + three - four;
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
    int one   = GET(img, x,     y);
    int two   = GET(img, x + w, y);
    int three = GET(img, x,     y + h / 3);
    int four  = GET(img, x + w, y + h / 3);
    int five  = GET(img, x,     y + h * 2 / 3);
    int six   = GET(img, x + w, y + h * 2 / 3);
    int seven = GET(img, x,     y + h);
    int eight = GET(img, x + w, y + h);
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
    int one   = GET(img, x,             y);
    int two   = GET(img, x + w / 3,     y);
    int three = GET(img, x + w * 2 / 3, y);
    int four  = GET(img, x + w,         y);
    int five  = GET(img, x,             y + h);
    int six   = GET(img, x + w / 3,     y + h);
    int seven = GET(img, x + w * 2 / 3, y + h);
    int eight = GET(img, x + w,         y + h);
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
    //    - 2 * (2) - 2 * (4) - 2 * (6) - 2 * (8)
    int one   = GET(img, x,         y);
    int two   = GET(img, x + w / 2, y);
    int three = GET(img, x + w,     y);
    int four  = GET(img, x,         y + h / 2);
    int five  = GET(img, x + w / 2, y + h / 2);
    int six   = GET(img, x + w,     y + h / 2);
    int seven = GET(img, x,         y + h);
    int eight = GET(img, x + w / 2, y + h);
    int nine  = GET(img, x + w,     y + h);
    int feature = 4 * five + one + three + seven + nine \
                  - 2 * two - 2 * four - 2 * six - 2 * eight;
    return feature;
  }
  default:
    fprintf(stderr, "unknown classifier id %d\n", id);
    fputs(__func__, stderr);
    exit(EXIT_FAILURE);
  }

#undef GET

}
