#include <chrono>
#include <iostream>
#include "../include/dbnet_base.hpp"
#include "../include/utils.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
int main(int argc, char** argv)
{
    std::string ini_path = "../config/dbnet.ini";
    std::string wts_path = "../models/trt_dbnet.onnx";
    std::string engine_path = "../models/dbnet.engine";
    const char* imgs_path = "../imgs";

    std::shared_ptr<TRTDBnetBase> dbnet = CreateDBnet();
    dbnet->init(ini_path);
    dbnet->load_model(wts_path, engine_path);

    std::vector<std::string> file_names;
    if (read_files_in_dir(imgs_path, file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    int fcount = 0;

    for (auto f : file_names) {
        std::vector<TRTDBnetResult> temp;
        fcount++;
        std::cout << fcount << "  " << f << std::endl;
        cv::Mat image = cv::imread("../imgs/" + f);
        cv::Mat src_img = image.clone();
        auto start = std::chrono::system_clock::now();
        temp = dbnet->detect(image);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(
                         (end - start))
                         .count()
                  << "ms" << std::endl;
        for (auto& one : temp) {
            std::cout << one.box_coordinates[0] << " " << one.box_coordinates[1]
                      << ",";
            std::cout << one.box_coordinates[2] << " " << one.box_coordinates[3]
                      << ",";
            std::cout << one.box_coordinates[4] << " " << one.box_coordinates[5]
                      << ",";
            std::cout << one.box_coordinates[6] << " " << one.box_coordinates[7]
                      << std::endl;
            std::cout << one.score << std::endl;
        }

        for (auto& one : temp) {
            cv::Point2f point1(one.box_coordinates[0], one.box_coordinates[1]);
            cv::Point2f point2(one.box_coordinates[2], one.box_coordinates[3]);
            cv::Point2f point3(one.box_coordinates[4], one.box_coordinates[5]);
            cv::Point2f point4(one.box_coordinates[6], one.box_coordinates[7]);

            cv::line(src_img, point1, point2, cv::Scalar(0, 0, 255), 2, 8);
            cv::line(src_img, point2, point3, cv::Scalar(0, 0, 255), 2, 8);
            cv::line(src_img, point3, point4, cv::Scalar(0, 0, 255), 2, 8);
            cv::line(src_img, point4, point1, cv::Scalar(0, 0, 255), 2, 8);
        }
        cv::imwrite("../outputs/" + f, src_img);
    }

    return 0;
}
