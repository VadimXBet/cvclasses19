/* FAST corner detector algorithm testing.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <catch2/catch.hpp>

#include "cvlib.hpp"

using namespace cvlib;

TEST_CASE("simple check", "[corner_detector_fast]")
{
    cv::Mat image(10, 10, CV_8UC1, cv::Scalar(0));
    auto fast = corner_detector_fast::create();
    SECTION("empty image")
    {
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(out.empty());
    }

    // \todo add 5 or more tests (SECTIONs)
    SECTION("solid image")
    {
        const auto image = cv::Mat(3, 3, CV_8UC1, cv::Scalar(127));
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(out.empty());
    }

    SECTION("angle (down)")
    {
        cv::Mat image = (cv::Mat_<uint8_t>(7, 7) <<
                127, 127, 127, 127, 127, 127, 127,
                127, 127, 127, 127, 127, 127, 127,
                127, 127, 127, 127, 127, 127, 127,
                127, 127, 127, 255, 127, 127, 127,
                127, 127, 255, 255, 127, 127, 127,
                127, 255, 255, 255, 127, 127, 127,
                255, 255, 255, 255, 127, 127, 127);
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(1 == out.size());
        REQUIRE(out[0].pt.x == 3);
        REQUIRE(out[0].pt.y == 3);
    }

    SECTION("angle (down1)")
    {
        cv::Mat image = (cv::Mat_<uint8_t>(7, 7) <<
                127, 127, 127, 127, 127, 127, 127,
                127, 127, 127, 127, 127, 127, 127,
                127, 127, 127, 127, 127, 127, 127,
                127, 127, 127, 255, 127, 127, 127,
                127, 127, 255, 255, 255, 127, 127,
                127, 255, 255, 255, 255, 255, 127,
                255, 255, 255, 255, 255, 255, 255);
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(1 == out.size());
        REQUIRE(out[0].pt.x == 3);
        REQUIRE(out[0].pt.y == 3);
    }

    SECTION("angle (left)")
    {
        cv::Mat image = (cv::Mat_<uint8_t>(7, 7) <<
                127, 127, 127, 127, 127, 127, 127,
                255, 127, 127, 127, 127, 127, 127,
                255, 255, 255, 127, 127, 127, 127,
                255, 255, 255, 255, 127, 127, 127,
                255, 255, 255, 127, 127, 127, 127,
                255, 127, 127, 127, 127, 127, 127,
                127, 127, 127, 127, 127, 127, 127);
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(1 == out.size());
        REQUIRE(out[0].pt.x == 3);
        REQUIRE(out[0].pt.y == 3);
    }

    SECTION("angle (up)")
    {
        cv::Mat image = (cv::Mat_<uint8_t>(7, 7) <<
                127, 127, 127, 255, 255, 255, 255,
                127, 127, 127, 255, 255, 255, 127,
                127, 127, 127, 255, 255, 127, 127,
                127, 127, 127, 255, 127, 127, 127,
                127, 127, 127, 127, 127, 127, 127,
                127, 127, 127, 127, 127, 127, 127,
                127, 127, 127, 127, 127, 127, 127);
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(1 == out.size());
        REQUIRE(out[0].pt.x == 3);
        REQUIRE(out[0].pt.y == 3);
    }
}
