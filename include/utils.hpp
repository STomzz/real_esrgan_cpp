#ifndef UTILS_H
#define UTILS_H


#include "opencv2/opencv.hpp"

class RealESRGANer
{
private:
    /* data */
public:
    cv::Mat img_input;
    int ROWS,COLS,CHANNELS,input_size,output_size;
    std::vector<float>input_data;
    // float *input_data_ptr;
    std::vector<float>output_data;
    std::vector<float>output_data_img;


    RealESRGANer(cv::Mat img);
    ~RealESRGANer();
    int pre_process(cv::Mat img);
    int process(std::string model_path);
    int post_process();
    
};

#endif