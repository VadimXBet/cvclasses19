/* Split and merge segmentation algorithm testing.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <catch2/catch.hpp>
#include "cvlib.hpp"

using namespace cvlib;

TEST_CASE("constant image", "[split_and_merge]")
{
    const cv::Mat image(100, 100, CV_8UC1, cv::Scalar{15});

    const auto res = split_and_merge(image, 1);
    REQUIRE(image.size() == res.size());
    REQUIRE(image.type() == res.type());
    REQUIRE(cv::Scalar(15) == cv::mean(res));
}

TEST_CASE("simple regions", "[split_and_merge]")
{
    SECTION("2x2")
    {
        const cv::Mat reference = (cv::Mat_<char>(2, 2) << 2, 2, 2, 2);
        cv::Mat image = (cv::Mat_<char>(2, 2) << 0, 1, 2, 3);
        auto res = split_and_merge(image, 10);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));

        res = split_and_merge(image, 1);
        REQUIRE(0 == cv::countNonZero(image - res));
    }

    SECTION("3x3")
    {
        // clang-format off
        const cv::Mat image = (cv::Mat_<char>(3, 3) <<
                0, 1, 2,
                3, 4, 5,
                6, 7, 8
        );
        const cv::Mat reference = (cv::Mat_<char>(3, 3) <<
                4, 4, 4,
                4, 4, 4,
                4, 4, 4
        );
        // clang-format on
        auto res = split_and_merge(image, 10);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));

        res = split_and_merge(image, 1);
        REQUIRE(0 == cv::countNonZero(image - res));
    }
}

TEST_CASE("compex regions", "[split_and_merge]")
{
    SECTION("2x2")
    {
        // clang-format off
        const cv::Mat image = (cv::Mat_<char>(2, 2) <<
                5, 7,
                6, 6
        );
        const cv::Mat reference = (cv::Mat_<char>(2, 2) <<
                6, 6,
                6, 6
        );
        // clang-format on
        auto res = split_and_merge(image, 10);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));

        res = split_and_merge(image, 1);
        REQUIRE(0 == cv::countNonZero(image - res));
    }

    SECTION("3x3")
    {
        // clang-format off
        const cv::Mat image = (cv::Mat_<char>(3, 3) <<
                0,  0,  0,
                0,  10, 11,
                0,  40, 40
        );
        const cv::Mat reference = (cv::Mat_<char>(3, 3) <<
                0,  0,  0,
                0,  25, 25,
                0,  25, 25
        );
        // clang-format on
        auto res = split_and_merge(image, 10);
        REQUIRE(image.size() == res.size());
        REQUIRE(0 == cv::countNonZero(reference - res));
        REQUIRE(image.type() == res.type());

        res = split_and_merge(image, 1);
        REQUIRE(0 == cv::countNonZero(image - res));
    }

    SECTION("4x4")
    {
        // clang-format off
        const cv::Mat image = (cv::Mat_<char>(4, 4) <<
                40, 43, 0, 3,
                41, 42, 6, 1,
                98, 92, 4, 2,
                96, 94, 5, 7
        );
        const cv::Mat reference = (cv::Mat_<char>(4, 4) <<
                42, 42, 2, 2,
                42, 42, 2, 2,
                95, 95, 4, 4,
                95, 95, 4, 4
        );
        // clang-format on
        auto res = split_and_merge(image, 1);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));

        res = split_and_merge(image, 1);
        REQUIRE(0 == cv::countNonZero(image - res));
    }
}
