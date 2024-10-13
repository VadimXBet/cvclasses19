/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <iostream>

namespace cvlib
{
void motion_segmentation::apply(cv::InputArray _image, cv::OutputArray _fgmask, double learningRate)
{
    cv::Mat image = _image.getMat();
    cv::Mat fgmask = _fgmask.getMat();

    bg_model_ = (1 - learningRate) * bg_model_ + learningRate * image;
    cv::absdiff(image, bg_model_, fgmask);
    threshold(fgmask, fgmask, 25, 128, cv::THRESH_BINARY);
    cv::cvtColor(fgmask, fgmask, cv::COLOR_BGR2GRAY);
	
    _fgmask.create(image.size(), CV_8UC3);
    _fgmask.assign(fgmask);
}
} // namespace cvlib
