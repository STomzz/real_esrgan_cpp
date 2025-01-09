#include "opencv2/opencv.hpp"
#include <iostream>
#include <sstream>
// #include "utils.hpp"
#include "img_reader.hpp"
#include "preprocessors.hpp"
#include "trtrunner.hpp"
#include "postprocessors.hpp"
// #include "NvInfer.h"



int main() {
    // cv::Mat img = cv::imread("/root/zst/Realesrgan/esrgan_cpp/data/00003.png");
    // std::clog<<"img type: "<<img.type()<<std::endl;//CV_8UC3
    // std::string model_path = "/root/zst/Realesrgan/esrgan_cpp/models/realesrgan-x4_2.engine";
    // RealESRGANer realesrganer(img);
    // int ret = realesrganer.pre_process(img);
    // ret = realesrganer.process(model_path);
    // ret = realesrganer.post_process();

    // if(!ret){
    //     std::cout<<"pocess successfully"<<std::endl;
    // }
    img_reader img_reader("/root/zst/Realesrgan/esrgan_orin/data/00003.png");
    preprocessors preprocessors(img_reader.img);
    preprocessors.pre_process();
    trtrunner trtrunner("/root/zst/Realesrgan/esrgan_orin/models/realesrgan-x4_2.engine",preprocessors.input_data);
    trtrunner.run();
    postprocessors postprocessors(trtrunner.output_data);
    postprocessors.post_process();
    return 0;
}