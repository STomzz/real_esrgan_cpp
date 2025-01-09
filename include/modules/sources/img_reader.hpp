#pragma once

#include <opencv2/core.hpp>

class img_reader {
public:
    cv::Mat img;

    img_reader(std::string img_path);
    ~img_reader();
    
};