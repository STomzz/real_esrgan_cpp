#pragma once

#include <opencv2/core.hpp>

class preprocessors {

public:
    
    cv::Mat img;
    std::vector<float> input_data;
    // int ROWS,COLS,CHANNELS;
    int input_size;
    

    preprocessors(cv::Mat img);
    bool pre_process();
    ~preprocessors();
};