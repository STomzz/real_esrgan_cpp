#pragma once
#include <string>
#include <vector>
#include "tensorrt_utils.hpp"

class trtrunner
{
private:
    
    int input_size, output_size;
    ICudaEngine* engine;
    IExecutionContext* context;
    cudaStream_t stream;
    void *buffers[2];
    /* data */
public:
    std::vector<float> input_data;
    std::vector<float> output_data;
    trtrunner(std::string model_path,std::vector<float> input_data);
    bool run();
    ~trtrunner();
};

