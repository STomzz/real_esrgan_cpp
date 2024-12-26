#include "utils.hpp"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <fstream>
#include <iostream>
#include "logging.h"
#include "cuda_runtime_api.h"


RealESRGANer::RealESRGANer(cv::Mat img){
    ROWS = img.rows;
    COLS = img.cols;
    CHANNELS = img.channels();
    input_size = ROWS*COLS*CHANNELS;
    
    input_data = std::vector<float>(input_size);
}
RealESRGANer::~RealESRGANer(){

}

int RealESRGANer::pre_process(cv::Mat img){
    int h_input = img.rows,w_input = img.cols;
    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);

    //h,w,c -> c,h,w
    for(int row = 0; row < ROWS; row++){
        for(int col = 0; col < COLS; col++){
            for(int channel = 0; channel < CHANNELS; channel++){
                //img(row, col, channel) -> input_data(channel, row, col)
                // int new_idx = channel*ROWS*COLS + row*COLS + col;
                int new_idx = (row*COLS+col)*CHANNELS+channel;
                // std::cout << "input_data[" <<  static_cast<float>(img.ptr<uchar>(row, col)[channel]) << std::endl;
                input_data[new_idx] = img.ptr<uchar>(row, col)[channel]/255.;
               
            }
        }
    }
    return 0;
}

int RealESRGANer::process(std::string model_path){

     std::string engine_path = model_path;
    std::ifstream file(engine_path, std::ios::binary);
    char*trt_model_stream = NULL;
    int size = 0;
    if(file.good()){
        file.seekg(0,file.end);
        size = file.tellg();
        file.seekg(0,file.beg);
        trt_model_stream = new char[size];
        assert(trt_model_stream);

        file.read(trt_model_stream,size);
        file.close();
    }


    Logger glogger;
    nvinfer1::IRuntime*runtime = nvinfer1::createInferRuntime(glogger);
    assert(runtime != nullptr);

    nvinfer1::ICudaEngine * engine = runtime->deserializeCudaEngine(trt_model_stream,size);
    assert(engine != nullptr);


    nvinfer1::IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trt_model_stream;

    void**data_buffer = new void*[2];
    int input_node_index = engine->getBindingIndex("input");
    cudaMalloc(&(data_buffer[input_node_index]), input_size * sizeof(float));

    int output_node_index = engine->getBindingIndex("output");
    cudaMalloc(&(data_buffer[output_node_index]), input_size*4 * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(data_buffer[0],input_data.data(),input_size*sizeof(float),cudaMemcpyHostToDevice,stream);


    context->enqueueV2(data_buffer,stream,nullptr);
    cudaMemcpyAsync(output_data.data(),data_buffer[output_node_index],input_size*4*sizeof(float),cudaMemcpyDeviceToHost,stream);
    return 0;
    return 0;
}
int RealESRGANer::post_process(){
    return 0;
}

