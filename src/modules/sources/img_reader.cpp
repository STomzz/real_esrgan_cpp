#include "img_reader.hpp"
#include "opencv2/opencv.hpp"
int ROWS, COLS, CHANNELS;
img_reader::img_reader(std::string img_path) {
    this->img = cv::imread(img_path);
    ROWS = img.rows;
    COLS = img.cols;
    CHANNELS = img.channels();
}

img_reader::~img_reader() {
}