#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"


using namespace std;
using namespace cv;

//public variables
cv::Ptr<cv::FeatureDetector> detector;
cv::Ptr<cv::DescriptorExtractor> extractor ;//= new cv::OrbDescriptorExtractor;
cv::Ptr<cv::DescriptorMatcher > matcher;// = new cv::BruteForceMatcher<cv::HammingLUT>;
BFMatcher _matcher;
KalmanFilter KF(4,2,0); 
Mat_<float> measurement(2,1);
Mat final_output;
 
//RobustMatcher class taken from OpenCV2 Computer Vision Application Programming Cookbook Ch 9
class RobustMatcher {
  private:
    // pointer to the feature point detector object
    cv::Ptr<cv::FeatureDetector> detector;
     // pointer to the feature descriptor extractor object
    cv::Ptr<cv::DescriptorExtractor> extractor;
    // pointer to the matcher object
     cv::Ptr<cv::DescriptorMatcher > matcher;
     float ratio; // max ratio between 1st and 2nd NN
     bool refineF; // if true will refine the F matrix
     double distance; // min distance to epipolar
     double confidence; // confidence level (probability)
  public:
     RobustMatcher() : ratio(0.65f), refineF(true),
                       confidence(0.99), distance(3.0) {
        // ORB is the default feature
        detector= new cv::OrbFeatureDetector();
        extractor= new cv::OrbDescriptorExtractor();
 		BFMatcher matcher(NORM_L2);
     //  matcher= new cv::BruteForceMatcher<cv::HammingLUT>;
     }

  // Set the feature detector
  void setFeatureDetector(
         cv::Ptr<cv::FeatureDetector>& detect) {
     detector= detect;
  }
  // Set the descriptor extractor
  void setDescriptorExtractor(
         cv::Ptr<cv::DescriptorExtractor>& desc) {
     extractor= desc;
  }
  // Set the matcher
  void setDescriptorMatcher(
         cv::Ptr<cv::DescriptorMatcher>& match) {
     matcher= match;
  }
  // Set confidence level
  void setConfidenceLevel(
         double conf) {
     confidence= conf;
  }
  //Set MinDistanceToEpipolar
  void setMinDistanceToEpipolar(
         double dist) {
     distance= dist;
  }
  //Set ratio
  void setRatio(
         float rat) {
     ratio= rat;
  }

  // Clear matches for which NN ratio is > than threshold
  // return the number of removed points
  // (corresponding entries being cleared,
  // i.e. size will be 0)
  int ratioTest(std::vector<std::vector<cv::DMatch> >
                                               &matches) {
    int removed=0;
      // for all matches
    for (std::vector<std::vector<cv::DMatch> >::iterator
             matchIterator= matches.begin();
         matchIterator!= matches.end(); ++matchIterator) {
           // if 2 NN has been identified
           if (matchIterator->size() > 1) {
               // check distance ratio
               if ((*matchIterator)[0].distance/
                   (*matchIterator)[1].distance > ratio) {
                  matchIterator->clear(); // remove match
                  removed++;
               }
           } else { // does not have 2 neighbours
               matchIterator->clear(); // remove match
               removed++;
           }
    }
    return removed;
  }

  // Insert symmetrical matches in symMatches vector
  void symmetryTest(
      const std::vector<std::vector<cv::DMatch> >& matches1,
      const std::vector<std::vector<cv::DMatch> >& matches2,
      std::vector<cv::DMatch>& symMatches) {
    // for all matches image 1 -> image 2
    for (std::vector<std::vector<cv::DMatch> >::
             const_iterator matchIterator1= matches1.begin();
         matchIterator1!= matches1.end(); ++matchIterator1) {
       // ignore deleted matches
       if (matchIterator1->size() < 2)
           continue;
       // for all matches image 2 -> image 1
       for (std::vector<std::vector<cv::DMatch> >::
          const_iterator matchIterator2= matches2.begin();
           matchIterator2!= matches2.end();
           ++matchIterator2) {
           // ignore deleted matches
           if (matchIterator2->size() < 2)
              continue;
           // Match symmetry test
           if ((*matchIterator1)[0].queryIdx ==
               (*matchIterator2)[0].trainIdx &&
               (*matchIterator2)[0].queryIdx ==
               (*matchIterator1)[0].trainIdx) {
               // add symmetrical match
                 symMatches.push_back(
                   cv::DMatch((*matchIterator1)[0].queryIdx,
                             (*matchIterator1)[0].trainIdx,
                             (*matchIterator1)[0].distance));
                 break; // next match in image 1 -> image 2
           }
       }
    }
  }

  // Identify good matches using RANSAC
  // Return fundemental matrix
  cv::Mat ransacTest(
      const std::vector<cv::DMatch>& matches,
      const std::vector<cv::KeyPoint>& keypoints1,
      const std::vector<cv::KeyPoint>& keypoints2,
      std::vector<cv::DMatch>& outMatches) {
   // Convert keypoints into Point2f
   std::vector<cv::Point2f> points1, points2;
   cv::Mat fundemental;
   for (std::vector<cv::DMatch>::
         const_iterator it= matches.begin();
       it!= matches.end(); ++it) {
       // Get the position of left keypoints
       float x= keypoints1[it->queryIdx].pt.x;
       float y= keypoints1[it->queryIdx].pt.y;
       points1.push_back(cv::Point2f(x,y));
       // Get the position of right keypoints
       x= keypoints2[it->trainIdx].pt.x;
       y= keypoints2[it->trainIdx].pt.y;
       points2.push_back(cv::Point2f(x,y));
    }
   // Compute F matrix using RANSAC
   std::vector<uchar> inliers(points1.size(),0);
   if (points1.size()>0&&points2.size()>0){
      cv::Mat fundemental= cv::findFundamentalMat(
         cv::Mat(points1),cv::Mat(points2), // matching points
          inliers,       // match status (inlier or outlier)
          CV_FM_RANSAC, // RANSAC method
          distance,      // distance to epipolar line
          confidence); // confidence probability
      // extract the surviving (inliers) matches
      std::vector<uchar>::const_iterator
                         itIn= inliers.begin();
      std::vector<cv::DMatch>::const_iterator
                         itM= matches.begin();
      // for all matches
      for ( ;itIn!= inliers.end(); ++itIn, ++itM) {
         if (*itIn) { // it is a valid match
             outMatches.push_back(*itM);
          }
       }
       if (refineF) {
       // The F matrix will be recomputed with
       // all accepted matches
          // Convert keypoints into Point2f
          // for final F computation
          points1.clear();
          points2.clear();
          for (std::vector<cv::DMatch>::
                 const_iterator it= outMatches.begin();
              it!= outMatches.end(); ++it) {
              // Get the position of left keypoints
              float x= keypoints1[it->queryIdx].pt.x;
              float y= keypoints1[it->queryIdx].pt.y;
              points1.push_back(cv::Point2f(x,y));
              // Get the position of right keypoints
              x= keypoints2[it->trainIdx].pt.x;
              y= keypoints2[it->trainIdx].pt.y;
              points2.push_back(cv::Point2f(x,y));
          }
          // Compute 8-point F from all accepted matches
          if (points1.size()>0&&points2.size()>0){
             fundemental= cv::findFundamentalMat(
                cv::Mat(points1),cv::Mat(points2), // matches
                CV_FM_8POINT); // 8-point method
          }
       }
    }
    return fundemental;
  }

  // Match feature points using symmetry test and RANSAC
  // returns fundemental matrix
  cv::Mat match(cv::Mat& image1,
                cv::Mat& image2, // input images
     // output matches and keypoints
     std::vector<cv::DMatch>& matches,
     std::vector<cv::KeyPoint>& keypoints1,
     std::vector<cv::KeyPoint>& keypoints2) {
   // 1a. Detection of the SURF features
   detector->detect(image1,keypoints1);
   detector->detect(image2,keypoints2);
   // 1b. Extraction of the SURF descriptors
   cv::Mat descriptors1, descriptors2;
   extractor->compute(image1,keypoints1,descriptors1);
   extractor->compute(image2,keypoints2,descriptors2);
   // 2. Match the two image descriptors
   // Construction of the matcher
   //cv::BruteForceMatcher<cv::L2<float>> matcher;
   // from image 1 to image 2
   // based on k nearest neighbours (with k=2)
   std::vector<std::vector<cv::DMatch> > matches1;
   matcher->knnMatch(descriptors1,descriptors2,
       matches1, // vector of matches (up to 2 per entry)
       2);        // return 2 nearest neighbours
    // from image 2 to image 1
    // based on k nearest neighbours (with k=2)
    std::vector<std::vector<cv::DMatch> > matches2;
    matcher->knnMatch(descriptors2,descriptors1,
       matches2, // vector of matches (up to 2 per entry)
       2);        // return 2 nearest neighbours
    // 3. Remove matches for which NN ratio is
    // > than threshold
    // clean image 1 -> image 2 matches
    int removed= ratioTest(matches1);
    // clean image 2 -> image 1 matches
    removed= ratioTest(matches2);
    // 4. Remove non-symmetrical matches
    std::vector<cv::DMatch> symMatches;
    symmetryTest(matches1,matches2,symMatches);
    // 5. Validate matches using RANSAC
    cv::Mat fundemental= ransacTest(symMatches,
                keypoints1, keypoints2, matches);
    // return the found fundemental matrix
    return fundemental;
  }
};
RobustMatcher rmatcher;




 void setRMatcher(){
    // set parameters

  int numKeyPoints = 2000;

  //Instantiate robust matcher



  //instantiate detector, extractor, matcher

  detector = new cv::ORB(numKeyPoints);
  extractor = new cv::OrbDescriptorExtractor;
  _matcher = BFMatcher(NORM_L2);
  matcher = &_matcher;//new cv::BFMatcher<cv::HammingLUT>();
  rmatcher.setFeatureDetector(detector);
  rmatcher.setDescriptorExtractor(extractor);
  rmatcher.setDescriptorMatcher(matcher);
}
void processFrames( Mat, Mat);

int FRAME_WIDTH = 427;//213;//854;
int FRAME_HEIGHT = 240;//120;//480;
double TARGET_HEIGHT = 0.8;
double TARGET_WIDTH = 0.8;
cv::Rect TARGET_RECTANGLE = cv::Rect((FRAME_WIDTH*(1-TARGET_WIDTH))/2,
                                    (FRAME_HEIGHT*(1-TARGET_HEIGHT))/2,
                                    FRAME_WIDTH*TARGET_WIDTH,
                                    FRAME_HEIGHT*TARGET_HEIGHT);

Point finalCenterPoint = cv::Point(0,0);

void KalmanInit(){
    //KF(4, 2, 0);
    //
    cout << "init kalman" <<endl;
    KF.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
    measurement.setTo(Scalar(0));
 
    // init...
    KF.statePre.at<float>(0) = finalCenterPoint.x;
    KF.statePre.at<float>(1) = finalCenterPoint.y;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KF.errorCovPost, Scalar::all(.1));
}
void NewWindows(){


  namedWindow("lastFrame",WINDOW_AUTOSIZE);
  namedWindow("debug",WINDOW_AUTOSIZE);
  namedWindow("frame",WINDOW_AUTOSIZE);
  namedWindow("final",WINDOW_AUTOSIZE);
}
void PoseWindows(){
  moveWindow("frame", 0,0);
  moveWindow("debug", 500,10);
  moveWindow("lastFrame", 0,500);
  moveWindow("final", 500,500);
}


void initTargetRect( int width, int height ){
  TARGET_RECTANGLE = cv::Rect((width*(1-TARGET_WIDTH))/2,
                                    (height*(1-TARGET_HEIGHT))/2,
                                    width*TARGET_WIDTH,
                                    height*TARGET_HEIGHT);
}

int main(int argc, char** argv)
{
  string videoPath = "../data/train.mp4";
  if(argc == 1){
    initTargetRect(FRAME_WIDTH, FRAME_HEIGHT);
  } else if(argc == 3){
    cout << atoi(argv[1]) << " " << atoi(argv[2]);
    initTargetRect(atoi(argv[1]), atoi(argv[2]));
    FRAME_WIDTH = atoi(argv[1]);
    FRAME_HEIGHT = atoi(argv[2]);
  } else if(argc == 4){
    initTargetRect(atoi(argv[1]), atoi(argv[2]));
    FRAME_WIDTH = atoi(argv[1]);
    FRAME_HEIGHT = atoi(argv[2]);
    videoPath = argv[3];
  }

  KalmanInit();
  NewWindows();

  cv::VideoCapture cap;
  if(videoPath.compare("") != 0){
    cap.open(videoPath);
  } else {
    cap.open(0);
  }
	cap.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

  cv::Mat frame;
	cv::Mat lastFrame;

// ============  
  //===========Inicjalizacja zapisu wideo
//==============

	Size frameSize(static_cast<int>(TARGET_RECTANGLE.width), static_cast<int>(TARGET_RECTANGLE.height));

  //zlap codec kamery
  int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));  //ex
VideoWriter oVideoWriter ("MyVideo.avi", CV_FOURCC('P','I','M','1'), 20, frameSize, true);


  int keyPressed = -1;
  int counter = 0;
  setRMatcher();
	while(keyPressed != 27){
  
  PoseWindows();
	
  	cap >> frame;
    if(frame.cols != FRAME_WIDTH){
      resize(frame, frame, Size(FRAME_WIDTH, FRAME_HEIGHT));
    }
    counter ++;

    cout << "nowa klatka " << frame.cols << " " << frame.rows << endl;

    
    if(!lastFrame.empty()){

      processFrames(lastFrame, frame);

      imshow("lastFrame", lastFrame);
      // save video
        cout << "nagrywanie " << final_output.cols << " x " << final_output.rows << endl; 
       oVideoWriter << final_output; //writer the frame into the file
    }

    imshow("frame", frame);
 
    if(counter > 15 && !frame.empty()){ //counter % 1 == 0){
  //       >
  //    \____/
      cout << "nowy frame" << endl;
      lastFrame = frame.clone();
     // lastFrame = lastFrame(TARGET_RECTANGLE);   /// ======= train.mp4 nie działa z tym
    }

		keyPressed = waitKey(33);

    //ustaw okienka
	}
  cap.release();
  oVideoWriter.release();


  return 0;
}

void processFrames( Mat lastFrame, Mat newFrame){

	//Load input image detect keypoints

	cv::Mat img1;
	std::vector<cv::KeyPoint> img1_keypoints;
	cv::Mat img1_descriptors;
	cv::Mat img2;
	std::vector<cv::KeyPoint> img2_keypoints;
	cv::Mat img2_descriptors;
	std::vector<cv::DMatch> matches;
	img1 = lastFrame;
	img2 = newFrame;
  cout<<img1.cols<<endl;
  cout<<img2.cols<<endl;
	rmatcher.match(img1, img2, matches, img1_keypoints, img2_keypoints);

	Mat debug;

	 cv::drawMatches(img1, img1_keypoints, img2, img2_keypoints, matches,
	 	 debug);


  std::vector<Point2f> obj;
  std::vector<Point2f> scene;
  for( int i = 0; i < matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( img1_keypoints[ matches[i].queryIdx ].pt );
    scene.push_back( img2_keypoints[ matches[i].trainIdx ].pt );
  }

  cout << "znalzl obj macze " << obj.size() << " " << scene.size() << endl;

  if(obj.size() > 4 && scene.size() > 4){
    Mat H = findHomography( cv::Mat(obj), cv::Mat(scene), CV_RANSAC );

    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img1.cols, 0 );
    obj_corners[2] = cvPoint( img1.cols, img1.rows ); obj_corners[3] = cvPoint( 0, img1.rows );
    std::vector<Point2f> scene_corners(4);

    perspectiveTransform( obj_corners, scene_corners, H);

    //Mat homography = img2.clone();
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( debug, scene_corners[0] + Point2f( img1.cols, 0), scene_corners[1] + Point2f( img1.cols, 0), Scalar(0, 255, 0), 4 );
    line( debug, scene_corners[1] + Point2f( img1.cols, 0), scene_corners[2] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );
    line( debug, scene_corners[2] + Point2f( img1.cols, 0), scene_corners[3] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );
    line( debug, scene_corners[3] + Point2f( img1.cols, 0), scene_corners[0] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );

    cv::Point2f center = cv::Point2f(0,0);
    for(int i = 0; i < scene_corners.size(); i ++){
      center += scene_corners[i];
    }
    center.x /= 4;
    center.y /= 4;
    //center += Point2f( img1.cols, 0);

    circle( debug, center, 20 , cv::Scalar(255,0,0), -1);


    //cout << "Kalman start" << center.x << " " << center.y << endl;

    if(center.x > FRAME_WIDTH || center.x < 0 ||
       center.y > FRAME_HEIGHT || center.y < 0){
      return;
    }

    
    // First predict, to update the internal statePre variable
    Mat prediction = KF.predict();
    Point predictPt(prediction.at<float>(0),prediction.at<float>(1));

    //cout << "Kalman start2" << center.x << " " << center.y << endl;

    float temp = float(center.x)/FRAME_WIDTH;
    //cout << "Kalman start3" << temp << endl;
                     
    measurement(0) = center.x;//(float)center.x/FRAME_WIDTH;

    measurement(1) = center.y;//(float)center.y/FRAME_HEIGHT;

    //cout << "Kalman mes " << measurement(0) << " " << measurement(1) << endl;


    Point measPt(measurement(0),measurement(1));
     
    // The "correct" phase that is going to use the predicted value and our measurement
    Mat estimated = KF.correct(measurement);
    Point statePt(estimated.at<float>(0),estimated.at<float>(1));


    cout << "Kalman mes " << statePt.x << " " << statePt.y << endl;

    //statePt.x *= FRAME_WIDTH;
    //statePt.y *= FRAME_HEIGHT;

    circle( debug, statePt, 10 , cv::Scalar(255,255,0), -1);
        imshow("debug", debug);

    if(statePt.x <TARGET_RECTANGLE.width/2) statePt.x = TARGET_RECTANGLE.width/2;
    if(statePt.y <TARGET_RECTANGLE.height/2) statePt.y = TARGET_RECTANGLE.height/2;
    if(statePt.x > TARGET_RECTANGLE.x*2 + TARGET_RECTANGLE.width/2) statePt.x = TARGET_RECTANGLE.x*2 + TARGET_RECTANGLE.width/2;
    if(statePt.y > TARGET_RECTANGLE.y*2 + TARGET_RECTANGLE.height/2) statePt.y = TARGET_RECTANGLE.y*2 + TARGET_RECTANGLE.height/2;

    final_output = img2.clone();
    final_output = final_output(cv::Rect(statePt.x - TARGET_RECTANGLE.width/2,
                                          statePt.y - TARGET_RECTANGLE.height/2,
                                          TARGET_RECTANGLE.width,
                                          TARGET_RECTANGLE.height));

    imshow("final", final_output);
    //imshow("homograhy", homography);
  }

  
}
