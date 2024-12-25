#include "include/utils.h"
#include "opencv2/opencv.hpp"


/*
*load model
*/
RealESRGANer::RealESRGANer(/* args */)
{

}

RealESRGANer::~RealESRGANer()
{
}

int RealESRGANer::pre_process(cv::Mat& img){
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    int ROWS = img.rows, COLS = img.cols, CHANNELS = img.channels();
    uchar data[CHANNELS*ROWS*COLS];
    //把hwc转为chw 像素点顺序变为[1,3,2] => [2,1,33]

    if(img.empty()){
        std::cerr<<"image is empty"<<std::endl;
        return -1;
    }
    // std::cout<<" row:"<<img_tmp.rows<<" cols:"<<img_tmp.cols<<" channels:"<<img_tmp.channels()<<std::endl;

    

    for(int row = 0; row < ROWS; row++){
        for(int col = 0; col < COLS; col++){
            for(int channel = 0; channel < CHANNELS; channel++){
                // data[channel*ROWS*COLS + row*COLS + col] = pdata[row*COLS*CHANNELS + col*CHANNELS + channel];
                data[channel*ROWS*COLS + row*COLS + col] = img.ptr<uchar>(row,col)[channel];
                // std::cout<<"  "<<img.ptr<uchar>(row,col)[channel]<<std::endl;
            }
        }
    }
    // std::clog<<" row:"<<img_tmp.rows<<" cols:"<<img_tmp.cols<<" channels:"<<img_tmp.channels()<<std::endl;

    //pre pad 
    uchar pad_data[CHANNELS*4*ROWS*4*COLS];
    memset(pad_data, 0, CHANNELS*ROWS*COLS*4);



   
    return 0;
}

int RealESRGANer::process(){
    return 0;
}

int RealESRGANer::post_process(){
    return 0;
}

int RealESRGANer::enhance(cv::Mat& img){
    cv::Mat rgbImage;
    cv::cvtColor(img, rgbImage, cv::COLOR_BGR2RGB);
    pre_process(rgbImage);//预处理
    return 0;

}
