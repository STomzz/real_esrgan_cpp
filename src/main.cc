#include "opencv2/opencv.hpp"
#include <iostream>
#include <sstream>
#include "utils.hpp"
// #include "NvInfer.h"

int main() {
    cv::Mat img = cv::imread("/root/zst/Realesrgan/esrgan_cpp/data/00003.png");
    std::string model_path = ""
    RealESRGANer realesrganer(img);
    realesrganer.pre_process(img);
    realesrganer.process()
    return 0;
}