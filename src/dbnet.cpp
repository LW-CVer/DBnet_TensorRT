#include "../include/dbnet.hpp"
#include <math.h>
#include <fstream>
#include "../include/INIReader.hpp"
#include "../include/common.hpp"
#include "../include/ini.hpp"
#include "../include/utils.hpp"

TRTDBnet::TRTDBnet()
    : m_buffers{nullptr, nullptr}, m_context(nullptr), m_runtime(nullptr),
      m_INPUT_BLOB_NAME("input"), m_OUTPUT_BLOB_NAME("output"),
      m_engine(nullptr)
{
}

TRTDBnet::~TRTDBnet()
{
    for (int i = 0; i < 2; i++) {
        if (m_buffers[i] != nullptr) {
            cudaFree(m_buffers[i]);
            m_buffers[i] = nullptr;
        }
    }
}

int TRTDBnet::init(const std::string& ini_path)
{
    INIReader reader(ini_path);
    m_gpu_index = reader.GetInteger("device", "gpu_index", 0);
    //论文默认0.3
    m_score_threshold = reader.GetReal("threshold", "score_threshold", 0.3);
    m_box_threshold = reader.GetReal("threshold", "box_threshold", 0.7);
    m_input_size = reader.GetInteger("tensorrt", "input_size", 736);
    m_use_fp16 = reader.GetBoolean("tensorrt", "fp16", false);
    m_equal_scale = reader.GetBoolean("tensorrt", "equal_scale", false);
    m_max_batchsize = reader.GetInteger("tensorrt", "max_batchsize", 1);
    m_max_candidates = reader.GetInteger("tensorrt", "max_candidates", 1000);
    m_min_size = reader.GetInteger("tensorrt", "min_size", 3);
    m_expand_ratio = reader.GetReal("tensorrt", "expand_ratio", 0.7);
    return 0;
}

int TRTDBnet::load_model(std::string& wts_file, std::string& engine_file)
{
    Logger_dbnet gLogger;
    cudaSetDevice(m_gpu_index);
    std::ifstream intrt(engine_file, std::ios::binary);
    if (intrt) {
        std::cout << "load local engine..." << engine_file << std::endl;
        m_runtime = nvinfer1::createInferRuntime(gLogger);
        intrt.seekg(0, std::ios::end);
        size_t length = intrt.tellg();
        intrt.seekg(0, std::ios::beg);
        std::vector<char> data(length);
        intrt.read(data.data(), length);
        m_engine = m_runtime->deserializeCudaEngine(data.data(), length);
        std::cout << "engine loaded." << std::endl;

    } else {
        std::cout << "create engine from wts..." << std::endl;
        auto builder = nvinfer1::createInferBuilder(gLogger);
        if (!builder) {
            return -1;
        }

        auto config = builder->createBuilderConfig();
        if (!config) {
            return -1;
        }

        m_engine = createEngine(m_max_batchsize, builder, config,
                                DataType::kFLOAT, wts_file);
        assert(m_engine != nullptr);

        nvinfer1::IHostMemory* engine_serialize = m_engine->serialize();
        std::ofstream out(engine_file.c_str(), std::ios::binary);
        out.write((char*)engine_serialize->data(), engine_serialize->size());

        std::cout << "serialize the engine to " << engine_file << std::endl;

        engine_serialize->destroy();
        config->destroy();
        builder->destroy();
    }
    m_context = m_engine->createExecutionContext();
    // m_engine->destroy();context还要使用时，对应的engine不能被destroy（）
    if (m_context == nullptr) {
        return -1;
    }
    /*
    for (int b = 0; b < m_engine->getNbBindings(); ++b) {
        if (m_engine->bindingIsInput(b))

            std::cout << "input:" << b << std::endl;
        else
            std::cout << "output:" << b << std::endl;
    }*/

    cudaStreamCreate(&m_stream);

    std::cout << "RT init done!" << std::endl;
    return 0;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* TRTDBnet::createEngine(unsigned int m_max_batchsize,
                                    IBuilder* builder, IBuilderConfig* config,
                                    DataType dt, std::string& wts_file)
{
    const auto explicitBatch =
        1U << static_cast<uint32_t>(
            NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name
    // m_INPUT_BLOB_NAME
    ITensor* data =
        network->addInput(m_INPUT_BLOB_NAME, dt, Dims4{1, 3, -1, -1});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(wts_file);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    /* ------ Resnet18 backbone------ */
    // Add convolution layer with 6 outputs and a 5x5 filter.
    IConvolutionLayer* conv1 = network->addConvolution(
        *data, 64, DimsHW{7, 7}, weightMap["backbone.conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{2, 2});
    conv1->setPadding(DimsHW{3, 3});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0),
                                      "backbone.bn1", 1e-5);
    IActivationLayer* relu1 =
        network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    IPoolingLayer* pool1 = network->addPooling(*relu1->getOutput(0),
                                               PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStride(DimsHW{2, 2});
    pool1->setPadding(DimsHW{1, 1});

    IActivationLayer* relu2 =
        basicBlock(network, weightMap, *pool1->getOutput(0), 64, 64, 1,
                   "backbone.layer1.0.");
    IActivationLayer* relu3 =
        basicBlock(network, weightMap, *relu2->getOutput(0), 64, 64, 1,
                   "backbone.layer1.1.");  // x2

    IActivationLayer* relu4 =
        basicBlock(network, weightMap, *relu3->getOutput(0), 64, 128, 2,
                   "backbone.layer2.0.");
    IActivationLayer* relu5 =
        basicBlock(network, weightMap, *relu4->getOutput(0), 128, 128, 1,
                   "backbone.layer2.1.");  // x3

    IActivationLayer* relu6 =
        basicBlock(network, weightMap, *relu5->getOutput(0), 128, 256, 2,
                   "backbone.layer3.0.");
    IActivationLayer* relu7 =
        basicBlock(network, weightMap, *relu6->getOutput(0), 256, 256, 1,
                   "backbone.layer3.1.");  // x4

    IActivationLayer* relu8 =
        basicBlock(network, weightMap, *relu7->getOutput(0), 256, 512, 2,
                   "backbone.layer4.0.");
    IActivationLayer* relu9 =
        basicBlock(network, weightMap, *relu8->getOutput(0), 512, 512, 1,
                   "backbone.layer4.1.");  // x5

    /* ------- FPN  neck ------- */
    ILayer* p5 = convBnLeaky(network, weightMap, *relu9->getOutput(0), 64, 1, 1,
                             1, "neck.reduce_conv_c5.conv",
                             ".bn");  // k=1 s = 1  p = k/2=1/2=0
    ILayer* c4_1 = convBnLeaky(network, weightMap, *relu7->getOutput(0), 64, 1,
                               1, 1, "neck.reduce_conv_c4.conv", ".bn");

    float* deval = reinterpret_cast<float*>(malloc(sizeof(float) * 64 * 2 * 2));
    for (int i = 0; i < 64 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts1{DataType::kFLOAT, deval, 64 * 2 * 2};
    IDeconvolutionLayer* p4_1 = network->addDeconvolutionNd(
        *p5->getOutput(0), 64, DimsHW{2, 2}, deconvwts1, emptywts);
    p4_1->setStrideNd(DimsHW{2, 2});
    p4_1->setNbGroups(64);
    weightMap["deconv1"] = deconvwts1;

    IElementWiseLayer* p4_add = network->addElementWise(
        *p4_1->getOutput(0), *c4_1->getOutput(0), ElementWiseOperation::kSUM);
    ILayer* p4 = convBnLeaky(network, weightMap, *p4_add->getOutput(0), 64, 3,
                             1, 1, "neck.smooth_p4.conv", ".bn");  // smooth
    ILayer* c3_1 = convBnLeaky(network, weightMap, *relu5->getOutput(0), 64, 1,
                               1, 1, "neck.reduce_conv_c3.conv", ".bn");

    Weights deconvwts2{DataType::kFLOAT, deval, 64 * 2 * 2};
    IDeconvolutionLayer* p3_1 = network->addDeconvolutionNd(
        *p4->getOutput(0), 64, DimsHW{2, 2}, deconvwts2, emptywts);
    p3_1->setStrideNd(DimsHW{2, 2});
    p3_1->setNbGroups(64);

    IElementWiseLayer* p3_add = network->addElementWise(
        *p3_1->getOutput(0), *c3_1->getOutput(0), ElementWiseOperation::kSUM);
    ILayer* p3 = convBnLeaky(network, weightMap, *p3_add->getOutput(0), 64, 3,
                             1, 1, "neck.smooth_p3.conv", ".bn");  // smooth
    ILayer* c2_1 = convBnLeaky(network, weightMap, *relu3->getOutput(0), 64, 1,
                               1, 1, "neck.reduce_conv_c2.conv", ".bn");

    Weights deconvwts3{DataType::kFLOAT, deval, 64 * 2 * 2};
    IDeconvolutionLayer* p2_1 = network->addDeconvolutionNd(
        *p3->getOutput(0), 64, DimsHW{2, 2}, deconvwts3, emptywts);
    p2_1->setStrideNd(DimsHW{2, 2});
    p2_1->setNbGroups(64);

    IElementWiseLayer* p2_add = network->addElementWise(
        *p2_1->getOutput(0), *c2_1->getOutput(0), ElementWiseOperation::kSUM);
    ILayer* p2 = convBnLeaky(network, weightMap, *p2_add->getOutput(0), 64, 3,
                             1, 1, "neck.smooth_p2.conv", ".bn");  // smooth

    Weights deconvwts4{DataType::kFLOAT, deval, 64 * 2 * 2};
    IDeconvolutionLayer* p3_up_p2 = network->addDeconvolutionNd(
        *p3->getOutput(0), 64, DimsHW{2, 2}, deconvwts4, emptywts);
    p3_up_p2->setStrideNd(DimsHW{2, 2});
    p3_up_p2->setNbGroups(64);

    float* deval2 =
        reinterpret_cast<float*>(malloc(sizeof(float) * 64 * 8 * 8));
    for (int i = 0; i < 64 * 8 * 8; i++) {
        deval2[i] = 1.0;
    }
    Weights deconvwts5{DataType::kFLOAT, deval2, 64 * 8 * 8};
    IDeconvolutionLayer* p4_up_p2 = network->addDeconvolutionNd(
        *p4->getOutput(0), 64, DimsHW{8, 8}, deconvwts5, emptywts);
    p4_up_p2->setPadding(DimsHW{2, 2});
    p4_up_p2->setStrideNd(DimsHW{4, 4});
    p4_up_p2->setNbGroups(64);
    weightMap["deconv2"] = deconvwts5;

    Weights deconvwts6{DataType::kFLOAT, deval2, 64 * 8 * 8};
    IDeconvolutionLayer* p5_up_p2 = network->addDeconvolutionNd(
        *p5->getOutput(0), 64, DimsHW{8, 8}, deconvwts6, emptywts);
    p5_up_p2->setStrideNd(DimsHW{8, 8});
    p5_up_p2->setNbGroups(64);

    // torch.cat([p2, p3, p4, p5], dim=1)
    ITensor* inputTensors[] = {p2->getOutput(0), p3_up_p2->getOutput(0),
                               p4_up_p2->getOutput(0), p5_up_p2->getOutput(0)};
    IConcatenationLayer* neck_cat = network->addConcatenation(inputTensors, 4);

    ILayer* neck_out =
        convBnLeaky(network, weightMap, *neck_cat->getOutput(0), 256, 3, 1, 1,
                    "neck.conv.0", ".1");  // smooth
    assert(neck_out);
    ILayer* binarize1 = convBnLeaky(network, weightMap, *neck_out->getOutput(0),
                                    64, 3, 1, 1, "head.binarize.0", ".1");  //
    Weights deconvwts7{DataType::kFLOAT, deval, 64 * 2 * 2};
    IDeconvolutionLayer* binarizeup = network->addDeconvolutionNd(
        *binarize1->getOutput(0), 64, DimsHW{2, 2}, deconvwts7, emptywts);
    binarizeup->setStrideNd(DimsHW{2, 2});
    binarizeup->setNbGroups(64);
    IScaleLayer* binarizebn1 = addBatchNorm2d(
        network, weightMap, *binarizeup->getOutput(0), "head.binarize.4", 1e-5);
    IActivationLayer* binarizerelu1 = network->addActivation(
        *binarizebn1->getOutput(0), ActivationType::kRELU);
    assert(binarizerelu1);

    Weights deconvwts8{DataType::kFLOAT, deval, 64 * 2 * 2};
    IDeconvolutionLayer* binarizeup2 = network->addDeconvolutionNd(
        *binarizerelu1->getOutput(0), 64, DimsHW{2, 2}, deconvwts8, emptywts);
    binarizeup2->setStrideNd(DimsHW{2, 2});
    binarizeup2->setNbGroups(64);

    IConvolutionLayer* binarize3 = network->addConvolution(
        *binarizeup2->getOutput(0), 1, DimsHW{3, 3},
        weightMap["head.binarize.7.weight"], weightMap["head.binarize.7.bias"]);
    assert(binarize3);
    binarize3->setStride(DimsHW{1, 1});
    binarize3->setPadding(DimsHW{1, 1});
    IActivationLayer* binarize4 = network->addActivation(
        *binarize3->getOutput(0), ActivationType::kSIGMOID);
    assert(binarize4);

    // threshold_maps = self.thresh(x)
    ILayer* thresh1 =
        convBnLeaky(network, weightMap, *neck_out->getOutput(0), 64, 3, 1, 1,
                    "head.thresh.0", ".1", false);  //
    Weights deconvwts9{DataType::kFLOAT, deval, 64 * 2 * 2};
    IDeconvolutionLayer* threshup = network->addDeconvolutionNd(
        *thresh1->getOutput(0), 64, DimsHW{2, 2}, deconvwts9, emptywts);
    threshup->setStrideNd(DimsHW{2, 2});
    threshup->setNbGroups(64);
    IConvolutionLayer* thresh2 = network->addConvolution(
        *threshup->getOutput(0), 64, DimsHW{3, 3},
        weightMap["head.thresh.3.1.weight"], weightMap["head.thresh.3.1.bias"]);
    assert(thresh2);
    thresh2->setStride(DimsHW{1, 1});
    thresh2->setPadding(DimsHW{1, 1});

    IScaleLayer* threshbn1 = addBatchNorm2d(
        network, weightMap, *thresh2->getOutput(0), "head.thresh.4", 1e-5);
    IActivationLayer* threshrelu1 =
        network->addActivation(*threshbn1->getOutput(0), ActivationType::kRELU);
    assert(threshrelu1);

    Weights deconvwts10{DataType::kFLOAT, deval, 64 * 2 * 2};
    IDeconvolutionLayer* threshup2 = network->addDeconvolutionNd(
        *threshrelu1->getOutput(0), 64, DimsHW{2, 2}, deconvwts10, emptywts);
    threshup2->setStrideNd(DimsHW{2, 2});
    threshup2->setNbGroups(64);
    IConvolutionLayer* thresh3 = network->addConvolution(
        *threshup2->getOutput(0), 1, DimsHW{3, 3},
        weightMap["head.thresh.6.1.weight"], weightMap["head.thresh.6.1.bias"]);
    assert(thresh3);
    thresh3->setStride(DimsHW{1, 1});
    thresh3->setPadding(DimsHW{1, 1});
    IActivationLayer* thresh4 = network->addActivation(
        *thresh3->getOutput(0), ActivationType::kSIGMOID);
    assert(thresh4);

    ITensor* inputTensors2[] = {binarize4->getOutput(0), thresh4->getOutput(0)};
    IConcatenationLayer* head_out = network->addConcatenation(inputTensors2, 2);

    // y = F.interpolate(y, size=(H, W))
    head_out->getOutput(0)->setName(m_OUTPUT_BLOB_NAME);
    network->markOutput(*head_out->getOutput(0));

    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(m_INPUT_BLOB_NAME, OptProfileSelector::kMIN,
                           Dims4(1, 3, 64, 64));
    profile->setDimensions(m_INPUT_BLOB_NAME, OptProfileSelector::kOPT,
                           Dims4(1, 3, 640, 640));
    profile->setDimensions(m_INPUT_BLOB_NAME, OptProfileSelector::kMAX,
                           Dims4(1, 3, 1920, 1920));
    config->addOptimizationProfile(profile);

    // Build engine
    builder->setMaxBatchSize(m_max_batchsize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
    if (m_use_fp16) {
        config->setFlag(BuilderFlag::kFP16);
    }

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;
}

void TRTDBnet::doInference(float* input, float* output, int h_scale,
                           int w_scale)
{
    // const ICudaEngine& engine = m_context->getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.

    // assert(engine.getNbBindings() == 2);
    // std::cout << 1.1 << std::endl;

    // In order to bind the buffers, we need to know the names of the input and
    // output tensors. Note that indices are guaranteed to be less than

    // IEngine::getNbBindings()
    // const int inputIndex = engine.getBindingIndex(m_INPUT_BLOB_NAME);
    // const int outputIndex = engine.getBindingIndex(m_OUTPUT_BLOB_NAME);
    // std::cout << h_scale << " " << w_scale << std::endl;

    m_context->setBindingDimensions(0, nvinfer1::Dims4{1, 3, h_scale, w_scale});
    // Create GPU buffers on device
    CHECK(cudaMalloc(&m_buffers[0], 3 * h_scale * w_scale * sizeof(float)));
    CHECK(cudaMalloc(&m_buffers[1], 2 * h_scale * w_scale * sizeof(float)));

    // DMA input batch data to device, infer on the batch asynchronously, and
    // DMA output back to host
    CHECK(cudaMemcpyAsync(m_buffers[0], input,
                          3 * h_scale * w_scale * sizeof(float),
                          cudaMemcpyHostToDevice, m_stream));
    m_context->enqueueV2(m_buffers, m_stream, nullptr);

    CHECK(cudaMemcpyAsync(output, m_buffers[1],
                          h_scale * w_scale * 2 * sizeof(float),
                          cudaMemcpyDeviceToHost, m_stream));
    cudaStreamSynchronize(m_stream);

    CHECK(cudaFree(m_buffers[0]));
    CHECK(cudaFree(m_buffers[1]));
}

std::vector<TRTDBnetResult> TRTDBnet::detect(cv::Mat& image)
{
    // auto start = std::chrono::system_clock::now();
    // std::vector<float> mean_value{0.406, 0.456, 0.485};  // BGR
    // std::vector<float> std_value{0.225, 0.224, 0.229};
    int ori_w = image.cols;
    int ori_h = image.rows;
    // auto start1 = std::chrono::system_clock::now();
    resize_img(image, m_input_size,m_equal_scale);
    // auto start2 = std::chrono::system_clock::now();
    float* input = new float[image.rows * image.cols * 3];
    float* output = new float[image.rows * image.cols * 2];
    // auto start3 = std::chrono::system_clock::now();

    float* input_ptr_pos = input;
    std::vector<cv::Mat> channel_splits;

    channel_splits.clear();
    for (size_t j = 0; j < 3; ++j) {
        channel_splits.emplace_back(image.rows, image.cols, CV_32FC1,
                                    (void*)input_ptr_pos);
        input_ptr_pos += image.rows * image.cols;
    }
    cv::split(image, channel_splits);
    //速度太慢
    /*
    int i = 0;
    //归一化,交换通道
    for (int row = 0; row < image.rows; ++row) {
        uchar* uc_pixel = image.data + row * image.step;
        for (int col = 0; col < image.cols; ++col) {
            input[i] = (uc_pixel[2] / 255.0 - mean_value[2]) / std_value[2];
            input[i + image.rows * image.cols] =
                (uc_pixel[1] / 255.0 - mean_value[1]) / std_value[1];
            input[i + 2 * image.rows * image.cols] =
                (uc_pixel[0] / 255.0 - mean_value[0]) / std_value[0];
            uc_pixel += 3;
            ++i;
        }
    }*/
    // auto end = std::chrono::system_clock::now();
    // std::cout <<
    // std::chrono::duration_cast<std::chrono::milliseconds>((start1
    // -start)).count()<< "ms" <<std::endl; std::cout <<
    // std::chrono::duration_cast<std::chrono::milliseconds>((start2
    // -start1)).count()<< "ms" <<std::endl; std::cout <<
    // std::chrono::duration_cast<std::chrono::milliseconds>((start3
    // -start2)).count()<< "ms" <<std::endl; std::cout <<
    // std::chrono::duration_cast<std::chrono::milliseconds>((end
    // -start3)).count()<< "ms" <<std::endl;
    doInference(input, output, image.rows, image.cols);

    cv::Mat map = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC1);
    for (int h = 0; h < image.rows; ++h) {
        uchar* ptr = map.ptr(h);
        for (int w = 0; w < image.cols; ++w) {
            ptr[w] = (output[h * image.cols + w] > m_score_threshold) ? 255 : 0;
        }
    }

    // 提取最小外接矩形
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarcy;
    cv::findContours(map, contours, hierarcy, CV_RETR_LIST,
                     CV_CHAIN_APPROX_SIMPLE);

    // std::vector<cv::Rect> boundRect(contours.size());
    std::vector<TRTDBnetResult> results;
    int choose_box =
        contours.size() < m_max_candidates ? contours.size() : m_max_candidates;
    cv::Point2f rect[4];
    for (int i = 0; i < choose_box; i++) {
        TRTDBnetResult one_result;

        cv::RotatedRect rotated_rect = cv::minAreaRect(contours[i]);
        if (!get_mini_boxes(rotated_rect, rect, m_min_size)) {
            continue;
        }
        float score = get_box_score(output, rect, image.cols, image.rows,
                                    m_score_threshold);
        if (score < m_box_threshold) {
            continue;
        }

        cv::RotatedRect expandbox = expandBox(rect, m_expand_ratio);
        expandbox.points(rect);
        if (!get_mini_boxes(expandbox, rect, m_min_size + 2)) {
            continue;
        }

        for (int j = 0; j < 4; j++) {
            //过滤因为扩大box导致的部分超界坐标
            if(rect[j].x<0){
                rect[j].x=0;
            }
            if(rect[j].x>(image.cols-1)){
                rect[j].x=image.cols-1;
            }
            if(rect[j].y<0){
                rect[j].y=0;
            }
            if(rect[j].y>(image.rows-1)){
                rect[j].y=image.rows-1;
            }
            one_result.box_coordinates[j * 2] =
                floor(rect[j].x / image.cols * ori_w);
            // floor(rect[j].x/scale);
            one_result.box_coordinates[j * 2 + 1] =
                floor(rect[j].y / image.rows * ori_h);
            // floor(rect[j].y/scale);
        }

        one_result.label = 0;
        one_result.score = score;
        results.push_back(one_result);
    }

    delete input;
    delete output;

    return results;
}

std::shared_ptr<TRTDBnetBase> CreateDBnet()
{
    return std::shared_ptr<TRTDBnetBase>(new TRTDBnet());
}