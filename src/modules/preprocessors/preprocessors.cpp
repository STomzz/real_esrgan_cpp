#include "preprocessors.hpp"
#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <fstream>
#include <iostream>
#include <vector>
#include "tensorrt_utils.hpp"
#include "opencv2/opencv.hpp"
extern int ROWS, COLS, CHANNELS;



preprocessors::preprocessors(cv::Mat img)
{   
    
    this->img = img;
    input_data.resize(ROWS*COLS*CHANNELS);
    
}

preprocessors::~preprocessors()
{
}

bool preprocessors::pre_process()
{   
     cv::cvtColor(img,img,cv::COLOR_BGR2RGB);


    //h,w,c -> c,h,w
    for(int row = 0; row < ROWS; row++){
        for(int col = 0; col < COLS; col++){
            for(int channel = 0; channel < CHANNELS; channel++){
                int new_idx = (channel*ROWS+row)*COLS+col;
                input_data[new_idx] = static_cast<float>(img.ptr<uchar>(row, col)[channel])/255.;
            }
        }
    }

    return true;
}