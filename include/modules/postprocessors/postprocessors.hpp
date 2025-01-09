#pragma once
#include <vector>

class postprocessors
{
private:

    /* data */
public:
    std::vector<float> output_data_img,output_data;
    postprocessors(std::vector<float> output_data);
    bool post_process();
    ~postprocessors();
};