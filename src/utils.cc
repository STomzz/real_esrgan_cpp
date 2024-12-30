#include "utils.hpp"
#include "cuda_runtime_api.h"
#include "NvInfer.h"
// #include "NvInferPlugin.h"
#include <fstream>
#include <iostream>
#include <vector>
#include "tensorrt_utils.hpp"

#ifdef DEBUG
  static Logger gLogger(ILogger::Severity::kINFO);
#else
  static Logger gLogger(ILogger::Severity::kERROR);
#endif


// 检查 CUDA 操作的宏
#define CHECK(status)                                                                          \
    do {                                                                                      \
        auto ret = (status);                                                                  \
        if (ret != 0) {                                                                       \
            std::cerr << "CUDA Error: " << cudaGetErrorString(ret) << " at line " << __LINE__; \
            exit(1);                                                                          \
        }                                                                                     \
    } while (0)



RealESRGANer::RealESRGANer(cv::Mat img){
    ROWS = img.rows;
    COLS = img.cols;
    CHANNELS = img.channels();
    input_size = ROWS*COLS*CHANNELS;
    output_size =  ROWS*4*COLS*4*CHANNELS;
    

    
    input_data.resize(input_size);
    output_data.resize(output_size);
    output_data_img.resize(output_size);
}
RealESRGANer::~RealESRGANer(){

}

int RealESRGANer::pre_process(cv::Mat img){
    // int IMAGE_HEIGHT = img.rows,IMAGE_WIDTH = img.cols,index = 0;

    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);


    //h,w,c -> c,h,w
    for(int row = 0; row < ROWS; row++){
        for(int col = 0; col < COLS; col++){
            for(int channel = 0; channel < CHANNELS; channel++){
                //img(row, col, channel) -> input_data(channel, row, col)
                // int new_idx = channel*ROWS*COLS + row*COLS + col;
                // int old_idx = (row*COLS+col)*CHANNELS+channel;
                int new_idx = (channel*ROWS+row)*COLS+col;
                // std::cout << "img" <<  static_cast<float>(img.ptr<uchar>(row, col)[channel]) << std::endl;
                input_data[new_idx] = static_cast<float>(img.ptr<uchar>(row, col)[channel])/255.;
                // input_data[new_idx] = static_cast<float>(img.data[old_idx]);
               
            }
        }
    }
    // img.convertTo(img,CV_32FC3,1.0/255);
    // int channelLength = IMAGE_WIDTH * IMAGE_HEIGHT;
    //     std::vector<cv::Mat> split_img = {
    //             cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC3, input_data_ptr + channelLength * (index + 2)),
    //             cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC3, input_data_ptr + channelLength * (index + 1)),
    //             cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC3, input_data_ptr + channelLength * index)
    //     };
    // cv::split(img, split_img);
    return 0;
}


int RealESRGANer::process(std::string model_path){
    auto engine = InitEngine(model_path, gLogger);

    auto nBinding_ = engine->getNbBindings();
    for(int i = 0; i < nBinding_; i++){
        auto tensor_name = engine->getBindingName(i);
        auto shape = engine->getTensorShape(tensor_name);
        auto datetype = engine->getBindingDataType(i);
        
        printf("Binding: %d \n\t tensor name->%s ,shape->%s ,datetype->%s ", i, tensor_name,
                shapeToString(shape).c_str(), dataTypeToString(datetype).c_str());
        }


    auto context = engine->createExecutionContext();
    if(!context){
        engine->destroy();
        throw std::runtime_error("failed to create TensorRT context");
    }

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // 分配 GPU 缓存
    void *buffers[2];
    CHECK(cudaMalloc(&buffers[0], input_size * sizeof(float)));  // 输入缓存
    CHECK(cudaMalloc(&buffers[1], output_size * sizeof(float))); // 输出缓存

    // 将输入拷贝到 GPU
    CHECK(cudaMemcpyAsync(buffers[0], input_data.data(), input_size * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 推理
    context->enqueueV2(buffers, stream, nullptr);

    // 将输出拷贝回主机
    CHECK(cudaMemcpyAsync(output_data.data(), buffers[1], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // 清理资源
    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));
    cudaStreamDestroy(stream);
    context->destroy();
    engine->destroy();

    return 0;
}




int RealESRGANer::post_process(){
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
    cv::imwrite("/root/zst/Realesrgan/esrgan_cpp/results/output_cuda.png",img_output);

    return 0;
}

