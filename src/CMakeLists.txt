# # 指定 CMake 的最小版本
# cmake_minimum_required(VERSION 3.10)

# # 设置项目名称和版本
# project(MyProject)

# 查找 OpenCV 版本
# set(OpenCV_DIR "C:/MyLibrary/OpenCV/opencv4_5_4_bin")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})



find_package(CUDA REQUIRED)


# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)




#添加头文件目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${TensorRT_INCLUDE_DIR}
    )
link_directories(${TensorRT_LIBRARY} ${TensorRT_PLUGIN_LIBRARY})
aux_source_directory(. SRC)#添加所有源文件




# 添加可执行文件
add_executable(${PROJECT_NAME}_exec ${SRC})



# 链接库
target_link_libraries(${PROJECT_NAME}_exec ${OpenCV_LIBS} nvinfer cudart)




