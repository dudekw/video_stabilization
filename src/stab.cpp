#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void processFrames( Mat, Mat);

int main(int argc, char** argv)
{

	cv::VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	cv::Mat frame;
	cv::Mat lastFrame;

	int keyPressed = -1;

	while(keyPressed != 27){

		lastFrame = frame;
		cap >> frame;

		if(!lastFrame.empty()){

			processFrames(lastFrame, frame);

			imshow("frame", frame);
			imshow("lastFrame", lastFrame);
		}

		keyPressed = waitKey(33);
	}

    cout<<"test"<< endl;
  return 0;
}

void processFrames( Mat lastFrame, Mat newFrame){

}