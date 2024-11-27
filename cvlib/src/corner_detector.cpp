/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>
#include <random>
#include <vector>
#include <algorithm>

namespace 
{
    uint8_t get_value(const cv::Mat& image, cv::Point point, cv::Point offset = cv::Point(0, 0)) 
    {
        return image.at<uint8_t>(point + offset);
    }

    int get_count_of_key_points(const cv::Mat& img, cv::Point point, const std::vector<cv::Point>& offsets)
    {
        int count = 0;
        const auto threshold = 5;
        std::vector<bool> answers_check1;
        std::vector<bool> answers_check2;
        for (int i = 0; i < offsets.size(); ++i)
        {
            const auto offset = offsets.at(i);
            const auto pixel = get_value(img, point);
            const auto offset_pixel = get_value(img, point, offset);
            const bool check1 = offset_pixel < pixel - threshold;
            const bool check2 = offset_pixel > pixel + threshold;
            if (offsets.size() == 4) 
            {
                const bool check = check1 || check2;
                if (check) ++count;
            }
            else
            {
                answers_check1.push_back(check1);
                answers_check2.push_back(check2);
            }
        }
        if (offsets.size() != 4)
        {
            int curr_count = 0;
            int n = answers_check1.size();
            for (int i = 1; i < 2*answers_check1.size(); ++i) 
            {
                bool answer1 = (answers_check1.at(i%n) && answers_check1.at((i-1)%n)); 
                bool answer2 = (answers_check2.at(i%n) && answers_check2.at((i-1)%n));
                curr_count = (answer1 || answer2) ? curr_count + 1 : 0;
                if (curr_count > count)
                    count = curr_count;
            }
        }
        return count;
    }

    cv::Point generate_point(int sigma)
    {
        // std::default_random_engine generator;
        std::random_device rd{};
        std::mt19937 generator{rd()};

        std::normal_distribution<float> distribution_x(0, sigma);
        int x = std::round(distribution_x(generator));

        std::normal_distribution<float> distribution_y(x, sigma);
        int y = std::round(distribution_y(generator));

        return {std::clamp(x, -sigma, sigma), std::clamp(y, -sigma, sigma)};
    }

    void make_point_pairs(std::vector<std::pair<cv::Point, cv::Point>> &pairs, const int desc_length, const int neighbourhood_size)
    {
        pairs.clear();
        const int halfsize = neighbourhood_size / 2;
        for (int i = 0; i < desc_length; i++)
            pairs.push_back(std::make_pair(generate_point(halfsize), generate_point(halfsize)));
    }
}

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
    keypoints.clear();
    // \todo implement FAST with minimal LOCs(lines of code), but keep code readable.
    const auto img_ = image.getMat();
    cv::Mat img;
    img_.copyTo(img);
    if (img.channels() == 3)
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    
    const auto offsets1 = std::vector<cv::Point>{cv::Point(0,  3), cv::Point(3,  0), cv::Point( 0,  -3), cv::Point(-3,  0)};
    const auto offsets2 = std::vector<cv::Point>{
        cv::Point(0, 3), cv::Point(1, 3), cv::Point(2, 2), cv::Point(3, 1),
        cv::Point(3, 0), cv::Point(3, -1), cv::Point(2, -2), cv::Point(1, -3),
        cv::Point(0, -3), cv::Point(-1, -3), cv::Point(-2, -2), cv::Point(-3, -1),
        cv::Point(-3, 0), cv::Point(-3, 1), cv::Point(-2, 2), cv::Point(-1, 3),
    };
    
    for (auto row = 3; row < img.rows - 3; ++row)
    {
        for (auto col = 3; col < img.cols - 3; ++col)
        {
            const auto center_coord = cv::Point(col, row);
            if (get_count_of_key_points(img, center_coord, offsets1) < 3) continue;
            if (get_count_of_key_points(img, center_coord, offsets2) >= 5)
                keypoints.push_back(cv::KeyPoint(center_coord, 3));
        }
    }
}

void corner_detector_fast::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    cv::Mat img;
    image.getMat().copyTo(img);
    if (img.channels() == 3)
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    std::vector<std::pair<cv::Point, cv::Point>> pairs;

    const int desc_length = 2;
    const int neighbourhood_size = 25;
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_8U);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);
    make_point_pairs(pairs, desc_length, neighbourhood_size);
    
    auto test = [&img](cv::Point kp, std::pair<cv::Point, cv::Point> p) { return img.at<uint8_t>(kp + p.first) < img.at<uint8_t>(kp + p.second); };

    auto ptr = reinterpret_cast<uint8_t*>(desc_mat.ptr());
    for (const auto& kp : keypoints)
    {
        for (int i = 0; i < desc_length; ++i)
        {
            uint8_t descriptor = 0;
            for (auto j = 0; j < pairs.size(); ++j)
                descriptor |= (test(kp.pt, pairs.at(j)) << (pairs.size() - 1 - j));
            *ptr = descriptor;
            ++ptr;
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray image, cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors, bool /*= false*/)
{
    // \todo implement me
    this->detect(image, keypoints);
    this->compute(image, keypoints, descriptors);
}
} // namespace cvlib
