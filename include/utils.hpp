#ifndef __TRT_DBNET_UTILS_HPP
#define __TRT_DBNET_UTILS_HPP
#include "opencv2/core.hpp"

bool get_mini_boxes(cv::RotatedRect& rotated_rect, cv::Point2f rect[],
                    int min_size);

float get_box_score(float* map, cv::Point2f rect[], int width, int height,
                    float threshold);

cv::RotatedRect expandBox(cv::Point2f rect[], float ratio = 1.5);

void resize_img(cv::Mat& In_Out_img, int shortsize = 640,
                bool equal_scale = false);

int read_files_in_dir(const char* p_dir_name,
                      std::vector<std::string>& file_names);
#endif