#include "utils.hpp"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <fstream>
#include <iostream>
#include "logging.h"
#include "cuda_runtime_api.h"


RealESRGANer::RealESRGANer(){

}
RealESRGANer::~RealESRGANer(){

}

int RealESRGANer::pre_process(){
    return 0;
}

int RealESRGANer::enhance(){
    return 0;
}
int RealESRGANer::post_process(){
    return 0;
}

