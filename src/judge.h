#ifndef __JUDGE_H_
#define __JUDGE_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

float overlapArea( cv::Rect &rc1, cv::Rect &rc2 );

float overlapIntersection( cv::Rect &box1, cv::Rect &box2 );

void judgeHandDisfection( std::vector<std::string> &classes, std::vector<std::string> &indices, std::vector<cv::Rect> &boxes );

void judgeHandTouchInfusion();


#endif
