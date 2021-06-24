#include <iostream>
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/dnn.hpp>

#include <vector>

#include "transport_udp.h"
#include "base64.h"

// ----------------- Initialize the parameters -----------------//
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image

std::vector<std::string> classes; // storage the names of the recgnized classes

std::vector<cv::Rect> positions; // storage the positions of the recgnized classes int every frame

pcs::Transport *udp;
struct sockaddr_in clientAddr1;
struct sockaddr_in clientAddr2;

// ------------- Get the classes' name from the .names file -----------//
void readClassName( std::string path, std::vector<std::string> &classes )
{
	std::ifstream ifs( path );
	
	std::string line;
	
	std::cout<<"classes: "<<std::endl;
	int count = 0;
	while( getline( ifs, line ) ){
		count ++;
		std::cout<<"class "<<count<<" : "<<line<<std::endl;
		classes.push_back( line );
	}
}

// ----------- Load the CNN network's weights file ------------//
bool loadModel( cv::dnn::Net &net, std::string modelConfig, std::string modelWeights )
{
	std::cout<<"read model configuration file from "<<modelConfig<<std::endl;
	std::cout<<"read model weights file from "<<modelWeights<<std::endl;

	net = cv::dnn::readNetFromDarknet( modelConfig, modelWeights );
	net.setPreferableBackend( cv::dnn::DNN_BACKEND_OPENCV );
	net.setPreferableTarget( cv::dnn::DNN_TARGET_CPU );

	return true;
}

void transmitIdentificatedResults(std::vector<std::string> &results)
{
	std::string objects = "";
	for( auto it: results ){
		objects += ( it + " " );	
	}
	int ret = udp->write(udp->getClientFd(), (unsigned char *)objects.c_str(), objects.size(), clientAddr2);
	
	if( ret > 0 )
        	std::cout<<"send "<<ret<<" bytes data"<<std::endl;
}

// ----------- Draw the predicted bounding box 绘出框 --------//
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
        //Draw a rectangle displaying the bounding box 绘制矩形
        cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 4);

        //Get the label for the class name and its confidence
        std::string label = cv::format("%.2f", conf);//分类标签及其置信度
        //若存在类别标签，读取对应的标签
	if (!classes.empty()){
                CV_Assert(classId < (int)classes.size());
                label = classes[classId] + ":" + label;
        }
	std::cout<<"classId: "<<classId<<", label: "<<label<<std::endl;
 
        //Display the label at the top of the bounding box
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = cv::max(top, labelSize.height);
        //绘制框上文字
        cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1.0f, cv::Scalar(0, 0, 255), 2);
}


// ------ Remove the bounding boxes with low confidence using non-maxima suppression ------//
void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs)
{
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
 
	std::vector<std::string> objects;

	for (size_t i = 0; i < outs.size(); ++i){
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols){
			int a = outs[i].cols;//中心坐标+框的宽高+置信度+分为各个类别分数=2+2+1+80
			int b = outs[i].rows;//框的个数507
			cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);//取当前框的第六列到最后一列，即该框被分为80个类别，各个类别的评分
			cv::Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);//找出最大评分的类别
			if (confidence > confThreshold){//置信度阈值
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;
 
				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
	}
 
	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);//框、置信度、置信度阈值、非极大值抑制阈值、指标（输出）
	for (size_t i = 0; i < indices.size(); ++i){
		int idx = indices[i];//框序号
		cv::Rect box = boxes[idx];//框的坐标（矩形区域）
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
		
		// added 
		objects.push_back( classes[ classIds[idx] ] );
	}
	
	transmitIdentificatedResults( objects );
}
 
 
// --------------- Get the names of the output layers --------------//
std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net)
{
	static std::vector<cv::String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		std::vector<int> outLayers = net.getUnconnectedOutLayers();
 
		//get the names of all the layers in the network
		std::vector<cv::String> layersNames = net.getLayerNames();
 
		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

// -------------- Init a Udp client to transmit the data ----------//
void initTransmit()
{
	udp = new pcs::TransportUDP();
	udp->initSocketClient();

	clientAddr1.sin_family = AF_INET;
	clientAddr1.sin_addr.s_addr = inet_addr("192.168.8.240");
	clientAddr1.sin_port = htons(2333);

	clientAddr2.sin_family = AF_INET;
        clientAddr2.sin_addr.s_addr = inet_addr("192.168.8.240");
        clientAddr2.sin_port = htons(2334);
}

// ------------- Transmit the frame to the server to showing ------//
void transmitAFrame( cv::Mat &src )
{
	cv::Mat dst;
	cv::resize( src, dst, cv::Size( 500, 500 ) );

	// encode
	std::vector<unsigned char> data_encode;
        std::vector<int> quality;
        quality.push_back( CV_IMWRITE_JPEG_QUALITY );
        quality.push_back( 50 );
        cv::imencode( ".jpg", dst, data_encode, quality );

	std::cout<<"size of the encode data: "<<data_encode.size()<<std::endl;

	char encodeArray[50000];

        int size = data_encode.size();
        for( size_t i = 0; i < size; ++ i ){
                encodeArray[i] = data_encode[i];
                //std::cout<<"encodeArray["<<i<<"] = "<<encodeArray[i]<<std::endl;
        }

        //------------ base64 encode -----------//      
        char outArray[60000];
        int encodeLen = base64_encode( (unsigned char *)encodeArray, size, outArray );
        std::cout<<"encoded length: "<<encodeLen<<std::endl;

	int ret = udp->write(udp->getClientFd(), (unsigned char *)outArray, encodeLen, clientAddr1);
        std::cout<<"send "<<ret<<" bytes data"<<std::endl;
}

int main( int argc, char **argv )
{
	std::cout<<"Program Begins ..."<<std::endl;
	
	// ------------ Init Transport ------------//
	initTransmit();

	// ------------- Open The Video File --------------//
	cv::String video = "2.mp4";
	cv::VideoCapture cap;

	if( !cap.open( video ) ){
		std::cout<<"Open the Video File Failed ..."<<std::endl;
		exit(-1);
	}
	else std::cout<<"Open the Video File Successfully ..."<<std::endl;

	// ------------------ Init the DNN ------------------//
	cv::dnn::Net net;

	readClassName( "./myData.names", classes );
	loadModel( net, "./my_yolov3.cfg", "./my_yolov3_10000.weights" );

	cv::namedWindow( "detector", 0 );
	cvResizeWindow( "detector", 900, 600 );
	cv::Mat frame, blob;
	while( 1 ){
		cap >> frame;
		
		cv::dnn::blobFromImage( frame, blob, 1 / 255.0f, cv::Size( 416, 416 ), cv::Scalar(0, 0, 0), true, false );// previous process of every frame 
	
		net.setInput( blob ); // input the picture of the dnn
	
		std::vector<cv::Mat> outs;
		net.forward( outs, getOutputsNames( net ) ); // forward broadcast

		postprocess( frame, outs ); // post process
		
		//--------------- Caculate the Frequncy ----------------//
		std::vector<double> layesTimes;	
		double freq = cv::getTickFrequency() / 1000;
		double t = net.getPerfProfile( layesTimes ) / freq;
		std::string label = cv::format( "Inference time for a frame: %.2f ms", t );
		cv::putText(frame, label, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255));

		imshow("detector", frame);// draw the results
		transmitAFrame( frame);

		cv::waitKey( 10 );
	}

	cap.release();
	return 0;
}

