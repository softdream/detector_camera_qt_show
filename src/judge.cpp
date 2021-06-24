#include "judge.h"
#include <algorithm>


float overlapArea( cv::Rect &rc1, cv::Rect &rc2 )
{
	cv::Point p1, p2;                 //p1为相交位置的左上角坐标，p2为相交位置的右下角坐标
    	p1.x = std::max(rc1.x, rc2.x);
    	p1.y = std::max(rc1.y, rc2.y);

    	p2.x = std::min(rc1.x +rc1.width, rc2.x +rc2.width);
    	p2.y = std::min(rc1.y +rc1.height, rc2.y +rc2.height);

    	float AJoin = 0;
    	if( p2.x > p1.x && p2.y > p1.y )            //判断是否相交
    	{
        	AJoin = (p2.x - p1.x)*(p2.y - p1.y);    //如果先交，求出相交面积
    	}
    	float A1 = rc1.width * rc1.height;
    	float A2 = rc2.width * rc2.height;
    	float AUnion = (A1 + A2 - AJoin);                 //两者组合的面积

    	if( AUnion > 0 )
        	return (AJoin/AUnion);                  //相交面积与组合面积的比例
    	else
       	 	return 0;

}

float overlapIntersection( cv::Rect &box1, cv::Rect &box2 )
{
	if (box1.x > box2.x + box2.width) { return 0.0; }
  	if (box1.y > box2.y + box2.height) { return 0.0; }
  	if (box1.x + box1.width < box2.x) { return 0.0; }
  	if (box1.y + box1.height < box2.y) { return 0.0; }
  
	float colInt =  std::min(box1.x + box1.width, box2.x + box2.width) - std::max(box1.x, box2.x);
  	float rowInt =  std::min(box1.y + box1.height, box2.y + box2.height) - std::max(box1.y, box2.y);
  	float intersection = colInt * rowInt;
  	float area1 = box1.width*box1.height;
  	float area2 = box2.width*box2.height;
  	
	return intersection / (area1 + area2 - intersection);
}
	
void judgeHandDisfection( std::vector<std::string> &classes, std::vector<std::string> &indices, std::vector<cv::Rect> &boxes )
{

}


