#ifndef HEADPOSEDETECTOR_H
#define HEADPOSEDETECTOR_H

#include <opencv2/opencv.hpp>

#include <ncnn/platform.h>
#include <ncnn/net.h>

#include <memory>

class HeadPoseDetector
{
public:
    HeadPoseDetector(HeadPoseDetector const&) = delete;
    HeadPoseDetector& operator=(HeadPoseDetector const&) = delete;

    static std::shared_ptr<HeadPoseDetector> instance()
    {
        static std::shared_ptr<HeadPoseDetector> s { new HeadPoseDetector };
        return s;
    }

private:
    HeadPoseDetector();

public:
    bool detect(const cv::Mat &rgb, /*out*/double &yaw, /*out*/double &pitch, /*out*/double &roll);

private:
    double __calc(ncnn::Extractor &ex, int id);

private:
    ncnn::Net m_net;

    std::unique_ptr<ncnn::Layer> m_softmaxPtr;

    bool m_initialized = false;
};

#endif // HEADPOSEDETECTOR_H

