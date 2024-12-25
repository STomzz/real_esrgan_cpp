#ifndef UTILS_H
#define UTILS_H


#include "opencv2/opencv.hpp"

class RealESRGANer
{
private:
    /* data */
public:
    cv::Mat img_original;


    RealESRGANer(/* args */);
    ~RealESRGANer();
    int pre_process(cv::Mat& img);
    int process();
    int post_process();
    int enhance(cv::Mat& img);
};

#endif