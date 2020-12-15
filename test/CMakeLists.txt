add_executable(test_dbnet test_dbnet.cpp)
target_link_libraries(test_dbnet ${TRT_DBNET_LIBRARY_NAME})
set(OPENCV_NEED_LIBS opencv_imgproc opencv_highgui)
target_link_libraries(test_dbnet ${OPENCV_NEED_LIBS})


