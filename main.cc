#include "opencv2/opencv.hpp"
#include <iostream>
#include <utils.h>



int main(){
    cv::Mat img = cv::imread("./data/000.png");
    
    
    if (img.empty()) {
    std::cout << "Failed to load image" << std::endl;
    return -1; // 退出程序
    }



    RealESRGANer realesrgan;
    int ret = realesrgan.enhance(img);
    std::cout<<"ret: "<<ret<<std::endl;


    return 0;
}