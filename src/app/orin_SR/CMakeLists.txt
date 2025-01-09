# # 指定 CMake 的最小版本
# cmake_minimum_required(VERSION 3.10)

# # 设置项目名称和版本
# project(MyProject)

# 查找 OpenCV 版本
# set(OpenCV_DIR "C:/MyLibrary/OpenCV/opencv4_5_4_bin")
# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})



# find_package(CUDA REQUIRED)


# # cuda
# include_directories(/usr/local/cuda/include)
# link_directories(/usr/local/cuda/lib64)
# # tensorrt
# include_directories(/usr/include/x86_64-linux-gnu/)
# link_directories(/usr/lib/x86_64-linux-gnu/)




#添加头文件目录
# include_directories(
#     ${CMAKE_CURRENT_SOURCE_DIR}
#     ${CMAKE_SOURCE_DIR}/include
#     ${TensorRT_INCLUDE_DIR}
#     )
link_directories(${TensorRT_LIBRARY} ${TensorRT_PLUGIN_LIBRARY})
# aux_source_directory(. SRC)#添加所有源文件

aux_source_directory(${PROJECT_SOURCE_DIR}/src/app/orin_SR SR_SRC_DIR)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/utils SR_SRC_DIR)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/modules/sources SR_SRC_DIR)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/modules/postprocessors SR_SRC_DIR)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/modules/preprocessors SR_SRC_DIR)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/modules/runners SR_SRC_DIR)








# 添加可执行文件
add_executable(orin_SR ${SR_SRC_DIR})



# 链接库
target_link_libraries(orin_SR ${OpenCV_LIBS} nvinfer cudart)




