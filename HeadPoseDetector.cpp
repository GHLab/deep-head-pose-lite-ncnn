#include "HeadPoseDetector.h"

#include "hopenet_lite.mem.h"

#define kInputSize 224
#define kOutputSize 66

using namespace cv;

HeadPoseDetector::HeadPoseDetector()
{
    const int paramResult = m_net.load_param(hopenet_lite_param_bin);
    const int modelResult = m_net.load_model(hopenet_lite_bin);

    m_initialized = paramResult > 0 && modelResult > 0;

    m_softmaxPtr.reset(ncnn::create_layer("Softmax"));
    ncnn::ParamDict pd;
    m_softmaxPtr->load_param(pd);
}

bool HeadPoseDetector::detect(const cv::Mat &rgb, /*out*/double &yaw, /*out*/double &pitch, /*out*/double &roll)
{
    if (!m_initialized || rgb.empty())
        return false;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, rgb.cols, rgb.rows, kInputSize, kInputSize);
    const float mean_vals[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
    const float norm_vals[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = m_net.create_extractor();
    ex.input(0, in);

    yaw = __calc(ex, 219);
    pitch = __calc(ex, 220);
    roll = __calc(ex, 221);

    return true;
}

double HeadPoseDetector::__calc(ncnn::Extractor &ex, int id)
{
    ncnn::Mat out;
    ex.extract(id, out);

    if (out.w * out.h * out.c != kOutputSize)
        return 0;

    float idxTensor[kOutputSize];
    for (int i = 0; i < kOutputSize; i++)
    {
        idxTensor[i] = i;
    }

    m_softmaxPtr->forward_inplace(out, m_net.opt);

    double result = 0;

    for (int i = 0; i < kOutputSize; i++)
    {
        result += (out[i] * idxTensor[i]);
    }

    return result * 3 - 99;
}
