#ifndef __DBNET_HPP__
#define __DBNET_HPP__
#include <cuda_runtime_api.h>
#include <iostream>
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "dbnet_base.hpp"
#include <chrono>
#define CHECK(status)                                          \
    do {                                                       \
        auto ret = (status);                                   \
        if (ret != 0) {                                        \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

class Logger_dbnet : public nvinfer1::ILogger
{
   public:
    Logger_dbnet(Severity severity = Severity::kINFO)
        : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the
        // reportable
        if (severity > reportableSeverity)
            return;

        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "UNKNOWN: ";
                break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};

class TRTDBnet : public TRTDBnetBase
{
   public:
    TRTDBnet();
    ~TRTDBnet() override;
    int init(const std::string& ini_path) override;
    int load_model(std::string& onnx_file, std::string& engine_file) override;
    std::vector<TRTDBnetResult> detect(cv::Mat& image_ori) override;

   private:
    nvinfer1::IExecutionContext* m_context;
    nvinfer1::IRuntime* m_runtime;
    nvinfer1::ICudaEngine* m_engine;
    cudaStream_t m_stream;

    void* m_buffers[2];

    const char* m_INPUT_BLOB_NAME;
    const char* m_OUTPUT_BLOB_NAME;

    int m_gpu_index;
    bool m_use_fp16;
    bool m_equal_scale;
    int m_max_batchsize;
    int m_max_candidates;
    int m_input_size;
    float m_score_threshold;
    float m_box_threshold;
    int m_min_size;
    float m_expand_ratio;

    // nvinfer1::ICudaEngine* createEngine(unsigned int maxBatchSize,
    //                                     nvinfer1::IBuilder* builder,
    //                                     nvinfer1::IBuilderConfig* config,
    //                                     nvinfer1::DataType dt,
    //                                     std::string& wts_path);
    void doInference(float* input, float* output, int h_scale, int w_scale);
};

#endif