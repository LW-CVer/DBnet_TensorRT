aux_source_directory(./ trt_dbnet_src)
add_library(${TRT_DBNET_LIBRARY_NAME} SHARED ${trt_dbnet_src})

target_link_libraries(${TRT_DBNET_LIBRARY_NAME} PRIVATE nvinfer)
target_link_libraries(${TRT_DBNET_LIBRARY_NAME} PRIVATE cudart)
target_link_libraries(${TRT_DBNET_LIBRARY_NAME} PRIVATE ${OpenCV_LIBS})