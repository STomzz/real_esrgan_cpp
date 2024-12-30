#include "opencv2/opencv.hpp"
#include <iostream>
#include <sstream>
#include "utils.hpp"
// #include "NvInfer.h"

int main() {
    cv::Mat img = cv::imread("/root/zst/Realesrgan/esrgan_cpp/data/00003.png");
    std::clog<<"img type: "<<img.type()<<std::endl;//CV_8UC3
    std::string model_path = "/root/zst/Realesrgan/esrgan_cpp/engine/realesrgan-x4_2.engine";
    RealESRGANer realesrganer(img);
    int ret = realesrganer.pre_process(img);
    ret = realesrganer.process(model_path);
    ret = realesrganer.post_process();

    if(!ret){
        std::cout<<"pocess successfully"<<std::endl;
    }
    return 0;
}