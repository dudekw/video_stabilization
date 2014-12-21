#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{

	cv::VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	cv::Mat frame;

	int keyPressed = -1;

	while(keyPressed != 27){

		cap >> frame;

		imshow("frame", frame);

		keyPressed = waitKey(33);
	}

    cout<<"test"<< endl;
  return 0;
}

