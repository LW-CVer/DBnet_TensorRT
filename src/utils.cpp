#include "../include/utils.hpp"
#include <dirent.h>
#include <math.h>
#include "../include/clipper.hpp"
#include "opencv2/opencv.hpp"

bool get_mini_boxes(cv::RotatedRect& rotated_rect, cv::Point2f rect[],
                    int min_size)
{

    cv::Point2f temp_rect[4];
    rotated_rect.points(temp_rect);
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            if (temp_rect[i].x > temp_rect[j].x) {
                cv::Point2f temp;
                temp = temp_rect[i];
                temp_rect[i] = temp_rect[j];
                temp_rect[j] = temp;
            }
        }
    }
    int index0 = 0;
    int index1 = 1;
    int index2 = 2;
    int index3 = 3;
    if (temp_rect[1].y > temp_rect[0].y) {
        index0 = 0;
        index3 = 1;
    } else {
        index0 = 1;
        index3 = 0;
    }
    if (temp_rect[3].y > temp_rect[2].y) {
        index1 = 2;
        index2 = 3;
    } else {
        index1 = 3;
        index2 = 2;
    }

    rect[0] = temp_rect[index0];
    rect[1] = temp_rect[index1];
    rect[2] = temp_rect[index2];
    rect[3] = temp_rect[index3];

    if (rotated_rect.size.width < min_size ||
        rotated_rect.size.height < min_size) {
        return false;
    } else {
        return true;
    }
}

float get_box_score(float* map, cv::Point2f rect[], int width, int height,
                    float threshold)
{

    int xmin = width - 1;
    int ymin = height - 1;
    int xmax = 0;
    int ymax = 0;

    for (int j = 0; j < 4; j++) {
        if (rect[j].x < xmin) {
            xmin = rect[j].x;
        }
        if (rect[j].y < ymin) {
            ymin = rect[j].y;
        }
        if (rect[j].x > xmax) {
            xmax = rect[j].x;
        }
        if (rect[j].y > ymax) {
            ymax = rect[j].y;
        }
    }
    float sum = 0;
    int num = 0;
    for (int i = ymin; i <= ymax; i++) {
        for (int j = xmin; j <= xmax; j++) {
            if (map[i * width + j] > threshold) {
                sum = sum + map[i * width + j];
                num++;
            }
        }
    }

    return sum / num;
}

cv::RotatedRect expandBox(cv::Point2f temp[], float ratio)
{
    ClipperLib::Path path = {
        {ClipperLib::cInt(temp[0].x), ClipperLib::cInt(temp[0].y)},
        {ClipperLib::cInt(temp[1].x), ClipperLib::cInt(temp[1].y)},
        {ClipperLib::cInt(temp[2].x), ClipperLib::cInt(temp[2].y)},
        {ClipperLib::cInt(temp[3].x), ClipperLib::cInt(temp[3].y)}};
    double area = ClipperLib::Area(path);
    double distance;
    double length = 0.0;
    for (int i = 0; i < 4; i++) {
        length = length + sqrtf(powf((temp[i].x - temp[(i + 1) % 4].x), 2) +
                                powf((temp[i].y - temp[(i + 1) % 4].y), 2));
    }

    distance = area * ratio / length;

    ClipperLib::ClipperOffset offset;
    offset.AddPath(path, ClipperLib::JoinType::jtRound,
                   ClipperLib::EndType::etClosedPolygon);
    ClipperLib::Paths paths;
    offset.Execute(paths, distance);

    std::vector<cv::Point> contour;
    for (int i = 0; i < paths[0].size(); i++) {
        contour.emplace_back(paths[0][i].X, paths[0][i].Y);
    }
    offset.Clear();
    return cv::minAreaRect(contour);
}

void resize_img(cv::Mat& In_Out_img, int shortsize, bool equal_scale)
{  //按shortsize为基础进行等比例缩放，最小为shortsize
    int w = In_Out_img.cols;
    int h = In_Out_img.rows;
    float scale = 0;
    if (w < h) {
        scale = ((float)shortsize) / w;
        h = int(std::round(scale * h / 32) * 32);
        // h = std::round(scale * h);
        w = shortsize;
    } else {
        scale = ((float)shortsize) / h;
        w = int(std::round(scale * w / 32) * 32);
        // w = std::round(scale * w);
        h = shortsize;
    }

    if (equal_scale) {
        cv::resize(In_Out_img, In_Out_img, cv::Size(shortsize, shortsize));
    } else {
        cv::resize(In_Out_img, In_Out_img, cv::Size(w, h));
    }

    cv::cvtColor(In_Out_img, In_Out_img, cv::COLOR_BGR2RGB);
    In_Out_img.convertTo(In_Out_img, CV_32FC3, 1.0 / 255);
    std::vector<float> mean_value{0.485, 0.456, 0.406};
    std::vector<float> std_value{0.229, 0.224, 0.225};
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(In_Out_img, rgbChannels);
    for (auto i = 0; i < rgbChannels.size(); i++) {
        rgbChannels[i].convertTo(rgbChannels[i], CV_32FC1, 1.0 / std_value[i],
                                 (0.0 - mean_value[i]) / std_value[i]);
    }
    cv::merge(rgbChannels, In_Out_img);
}

int read_files_in_dir(const char* p_dir_name,
                      std::vector<std::string>& file_names)
{
    DIR* p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            // std::string cur_file_name(p_dir_name);
            // cur_file_name += "/";
            // cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }
    closedir(p_dir);
    return 0;
}
