#include <assert.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include "adaboost.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void usage(void) {
  cout << "Usage: ./face-detection <path/to/image>" << endl;
  exit(0);
}

void integral_image(vector<int> &img) {
  // FIXME: handle other image sizes
  int size = (img.size() == 16 * 16) ? 16 : 24;
  // per row
  for (int j = 0; j < size; j++) {
    // cumulative sum per row (but k is column number)
    for (int k = 1; k < size; k++) {
      img[j * size + k] += img[j * size + k - 1];
    }
  }
  // per column
  for (int j = 0; j < size; j++) {
    // cumulative sum per column (but k is row number)
    for (int k = 1; k < size; k++) {
      img[k * size + j] += img[(k - 1) * size + j];
    }
  }
}

vector<square> find_face(Mat &image,
                         strong_classifier &classifier,
                         unsigned min_size = img_size,
                         unsigned max_size = 0,
                         unsigned move_delta = 5,
                         unsigned steps = 10) {

  // the region of found faces
  vector<square> found;

  // handle parameters
  if (min_size < img_size) min_size = img_size;
  int w = image.cols, h = image.rows;
  if (max_size == 0) max_size = min(w, h);
  // min_size <= face_size <= max_size

  // scale down image: min_size --> img_size
  int delta_size = (max_size - min_size) / steps;
  // give it a small delta
  if (delta_size == 0) delta_size++;

  // give a small window move distance
  if (move_delta == 0) move_delta = 1;

  for (int size = min_size; size <= max_size; size += delta_size) {

    // rescale image
    Mat scaled;
    double scale = (double)img_size / size;
    resize(image, scaled, Size(), scale, scale);

    // window size is img_size x img_size
    for (int i = 0; i <= scaled.cols - img_size; i += move_delta) {
      for (int j = 0; j <= scaled.rows - img_size; j += move_delta) {
        // crop scaled
        vector<int> win;
        Rect region(i, j, img_size, img_size);
        Mat cropped;
        scaled(region).copyTo(cropped);
        cropped.reshape(1, img_size * img_size).copyTo(win);
        // integral scaled for the window
        integral_image(win);
        if (classifier.H(win) == 1) {
          square sq = {(int)(i / scale), (int)(j / scale), size};
          found.push_back(sq);
          // cout << "FOUND: " << sq.x << " " << sq.y << " " << sq.size << endl;
        }
      }
    }
  }

  return found;
}

int main(int argc, char **argv) {

  if (argc != 2) {
    usage();
  }
  char *filename = argv[1];

  Mat image;
  image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

  if (!image.data) {
    cout << "Could not open or find the image" << endl ;
    exit(EXIT_FAILURE);
  }

  vector<weak_classifier> wc = load_array<weak_classifier>("data/strong_classifier.dat");
  strong_classifier sc(wc);

  vector<square> found = find_face(image, sc, 120, 200, 20, 5);

  for (int i = 0; i < found.size(); i++) {
    cout << found[i].x << " " << found[i].y << " " << found[i].size << endl;
  }

  // namedWindow( "Display window", CV_WINDOW_AUTOSIZE );
  // imshow( "Display window", image );
  // waitKey(0);

  return 0;
}
