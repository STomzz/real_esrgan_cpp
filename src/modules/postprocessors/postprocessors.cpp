#include "postprocessors.hpp"
#include "opencv2/opencv.hpp"

extern int ROWS, COLS, CHANNELS;

postprocessors::postprocessors(std::vector<float> output_data)
{
    this->output_data = output_data;
    this->output_data_img.resize(ROWS*4*COLS*4*CHANNELS);
}

postprocessors::~postprocessors()
{
}

bool postprocessors::post_process()
{
     //c,h,w -> h,w,c
    int ROWS_SCALE = ROWS*4,COLS_SCALE = COLS*4;
    for(int row = 0; row < ROWS_SCALE; row++){
        for(int col = 0; col < COLS_SCALE; col++){
            for(int channel = 0; channel < CHANNELS; channel++){
                //img(row, col, channel) -> input_data(channel, row, col)
                // int new_idx = channel*ROWS*COLS + row*COLS + col;
                int old_idx = (row*COLS_SCALE+col)*CHANNELS+channel;
                int new_idx = (channel*ROWS_SCALE+row)*COLS_SCALE+col;
                // std::cout << "img" <<  static_cast<float>(img.ptr<uchar>(row, col)[channel]) << std::endl;
                output_data_img[old_idx] = static_cast<float>(output_data[new_idx])*255;
                // input_data[new_idx] = static_cast<float>(img.data[old_idx]);
               
            }
        }
    }

    cv::Mat img_output(ROWS*4,COLS*4,CV_32FC3,output_data_img.data());
    cv::imwrite("/root/zst/Realesrgan/esrgan_orin/results/output_cuda.png",img_output);

    return 0;
}