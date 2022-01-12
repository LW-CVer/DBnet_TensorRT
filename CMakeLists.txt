cmake_minimum_required(VERSION 3.10)

if (NOT DEFINED PROJECT_NAME)
    project(TRTDBnet)
endif()


set(CMAKE_CXX_STANDARD 11)
message(\n ---${CMAKE_CURRENT_SOURCE_DIR}---)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# opencv相关
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# cuda相关
if (NOT DEFINED CUDA_LINK_LIBRARIES_KEYWORD)
    set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
endif ()

if (NOT DEFINED CUDA_ROOT_DIR)
    set(CUDA_ROOT_DIR /usr/local/cuda)
endif ()
message(CUDA_ROOT_DIR: ${CUDA_ROOT_DIR})

include_directories(${CUDA_ROOT_DIR}/include)
link_directories(${CUDA_ROOT_DIR}/lib64)

# TensorRT相关
# 如果上层未定义TRT_ROOT，就定义，解决路径问题
if (NOT DEFINED TRT_ROOT_DIR)
    # 这个路径是本工程本地测试路径
    set(TRT_ROOT_DIR /usr/local/TensorRT-7.1.3.4/)
endif ()
message(TRT_ROOT_DIR: ${TRT_ROOT_DIR})
include_directories(${TRT_ROOT_DIR}/include)
link_directories(${TRT_ROOT_DIR}/lib)

# project相关
if (NOT DEFINED TRT_DBNET_LIBRARY_NAME)
    set(TRT_DBNET_LIBRARY_NAME trt_dbnet)
endif ()

add_subdirectory(src)
if (${PROJECT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    add_subdirectory(test)
endif ()
