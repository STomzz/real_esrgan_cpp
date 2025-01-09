# 指定 CMake 的最小版本
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和版本
project(MyProject)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})



find_package(CUDA REQUIRED)


# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)
include_directories(${TensorRT_INCLUDE_DIR})


#添加所有自定义头文件
#定义头文件所在目录
set(MY_HPP_DIR ${CMAKE_SOURCE_DIR}/include/modules)
#遍历所有目录
file(GLOB MY_HPP_FILES ${MY_HPP_DIR}/*)
foreach(SUBDIR ${MY_HPP_FILES})
    if(IS_DIRECTORY ${SUBDIR})
        include_directories(${SUBDIR})
    endif()
endforeach()

include_directories(${CMAKE_SOURCE_DIR}/include/utils)







#添加子目录
add_subdirectory(./src/app)


