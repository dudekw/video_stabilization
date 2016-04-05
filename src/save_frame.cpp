  #include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include <stdarg.h>
using namespace std;
using namespace cv;
int main(int argc, char** argv)
{


    VideoCapture capture("MyVideo_bez_v.avi"); 
    VideoCapture captureYT("stabilization_YT.avi"); 
    namedWindow("compare");
    	   	Mat frameYT;
    	captureYT>>frameYT;
    	    	Mat frame;
    	    	frame.size() = frameYT.size();
    	capture>>frame;
Size sz1 = frame.size();
    Size sz2 = frameYT.size();
    Mat im3;
    im3=Mat::zeros(sz2.height, sz1.width+sz2.width, CV_8UC3);
	// ============  
  //===========Inicjalizacja zapisu wideo
//==============
	Size frameSize(static_cast<int>(frame.cols), static_cast<int>(frame.rows));
	Size frameSizeYT(static_cast<int>(frameYT.cols), static_cast<int>(frameYT.rows));
		Size frameSizeim3(static_cast<int>(im3.cols), static_cast<int>(im3.rows));

  //zlap codec kamery
  int ex = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));  //ex
VideoWriter oVideoWriter ("compare_moj.avi", CV_FOURCC('P','I','M','1'), 20, frameSize, true);
VideoWriter oVideoWriterYT("compare_YT.avi", CV_FOURCC('P','I','M','1'), 20, frameSizeYT, true);
VideoWriter oVideoWriterim3("compare.avi", CV_FOURCC('P','I','M','1'), 20, frameSizeim3, true);
    for(;;) {
    capture >> frame;

 	circle( frame, Point2f(frame.cols/2,frame.rows/2), 20 , cv::Scalar(255,0,0), 2);
  	circle( frame, Point2f(frame.cols/2,frame.rows/2), 4 , cv::Scalar(255,0,0), -1);
      // save video
        cout << "nagrywanie " << frame.cols << " x " << frame.rows << endl; 
       oVideoWriter << frame; //writer the frame into the file
   // imshow("my_window", frame);
       
    captureYT >> frameYT;

 	circle( frameYT, Point2f(frameYT.cols/2,frameYT.rows/2), 20 , cv::Scalar(255,0,0), 2);
  	circle( frameYT, Point2f(frameYT.cols/2,frameYT.rows/2), 4 , cv::Scalar(255,0,0), -1);
      // save video
        cout << "nagrywanie2 " << frameYT.cols << " x " << frameYT.rows << endl; 
       oVideoWriterYT<< frameYT; //writer the frame into the file
   
    Size sz1 = frame.size();
    Size sz2 = frameYT.size();
    Mat im3;
    im3=Mat::zeros(sz2.height, sz1.width+sz2.width, CV_8UC3);
    Mat left(im3, Rect(0, sz2.height/2 - sz1.height/2, sz1.width, sz1.height));
    frame.copyTo(left);
    Mat right(im3, Rect(sz1.width, 0, sz2.width, sz2.height));
    frameYT.copyTo(right);
    imshow("compare", im3);
       oVideoWriterim3<< im3; //writer the frame into the file

    if(cv::waitKey(30) >= 0) break;

}
}