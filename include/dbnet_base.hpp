#ifndef __DBNET_BASE_HPP__
#define __DENET_BASE_HPP__
#include <memory>
#include <string>
#include <vector>
#include "opencv2/core/mat.hpp"
struct TRTDBnetResult
{
    int label;
    int box_coordinates[8];
    float score;
};

class TRTDBnetBase
{
   public:
    virtual ~TRTDBnetBase() = default;
    virtual int init(const std::string& ini_path) = 0;
    virtual int load_model(std::string& onnx_file, std::string& engine_file) = 0;
    virtual std::vector<TRTDBnetResult> detect(cv::Mat& image) = 0;
};

std::shared_ptr<TRTDBnetBase> CreateDBnet();
#endif