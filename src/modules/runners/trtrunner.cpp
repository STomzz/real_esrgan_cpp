#include "trtrunner.hpp"
// #include "utils.hpp"
#include "cuda_runtime_api.h"
#include "NvInfer.h"
// #include "NvInferPlugin.h"
#include <fstream>
#include <iostream>
#include <vector>
#include "tensorrt_utils.hpp"

extern int ROWS, COLS, CHANNELS;

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


trtrunner::trtrunner(std::string model_path,std::vector<float> input_data)
{ 
    this->input_size = ROWS*COLS*CHANNELS;
    this->output_size = input_size*16;
    this->input_data = input_data;
    output_data.resize(output_size);
    
    this->engine = InitEngine(model_path, gLogger);

    auto nBinding_ = engine->getNbBindings();
    for(int i = 0; i < nBinding_; i++){
        auto tensor_name = engine->getBindingName(i);
        auto shape = engine->getTensorShape(tensor_name);
        auto datetype = engine->getBindingDataType(i);
        
        printf("Binding: %d \n\t tensor name->%s ,shape->%s ,datetype->%s ", i, tensor_name,
                shapeToString(shape).c_str(), dataTypeToString(datetype).c_str());
        }


    this->context = engine->createExecutionContext();
    if(!context){
        engine->destroy();
        throw std::runtime_error("failed to create TensorRT context");
    }

}

trtrunner::~trtrunner()
{
    // 清理资源
    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));
    cudaStreamDestroy(stream);
    context->destroy();
    engine->destroy();
}


bool trtrunner::run()
{

    // cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // 分配 GPU 缓存
    // void *buffers[2];
    CHECK(cudaMalloc(&buffers[0], input_size * sizeof(float)));  // 输入缓存
    CHECK(cudaMalloc(&buffers[1], output_size * sizeof(float))); // 输出缓存

    // 将输入拷贝到 GPU
    CHECK(cudaMemcpyAsync(buffers[0], input_data.data(), input_size * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 推理
    context->enqueueV2(buffers, stream, nullptr);

    // 将输出拷贝回主机
    CHECK(cudaMemcpyAsync(output_data.data(), buffers[1], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    
    return true;
}
