#ifndef UTILS_H
#define UTILS_H


#include "opencv2/opencv.hpp"

class RealESRGANer
{
private:
    /* data */
public:
    cv::Mat img_input;
    int ROWS,COLS,CHANNELS;
    std::vector<float>input_data;
    int input_size;
    std::vector<float>output_data;


    RealESRGANer(cv::Mat img);
    ~RealESRGANer();
    int pre_process(cv::Mat img);
    int process(std::string model_path);
    int post_process();
    
};

#endif