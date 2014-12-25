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

  int numKeyPoints = 1500;

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

int main(int argc, char** argv)
{

	cv::VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	cv::Mat frame;
	cv::Mat lastFrame;

	int keyPressed = -1;
  int counter = 0;
  setRMatcher();
	while(keyPressed != 27){

		cap >> frame;
    counter ++;

    if(counter % 100 == 0){
      cout << "nowy frame" << endl;
      lastFrame = frame.clone();
    }
    if(!lastFrame.empty()){

      processFrames(lastFrame, frame);

      imshow("lastFrame", lastFrame);
    }

    imshow("frame", frame);

		keyPressed = waitKey(33);
	}
  cap.release();
    cout<<"test"<< endl;
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
	rmatcher.match(img1, img2, matches, img1_keypoints, img2_keypoints);

	Mat debug;

	cv::drawMatches(img1, img1_keypoints, img2, img2_keypoints, matches,
		 debug);

	imshow("debug", debug);

  std::vector<Point2f> obj;
  std::vector<Point2f> scene;

  for( int i = 0; i < matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( img1_keypoints[ matches[i].queryIdx ].pt );
    scene.push_back( img2_keypoints[ matches[i].trainIdx ].pt );
  }

  Mat H = findHomography( cv::Mat(obj), cv::Mat(scene), CV_RANSAC );

  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img1.cols, 0 );
  obj_corners[2] = cvPoint( img1.cols, img1.rows ); obj_corners[3] = cvPoint( 0, img1.rows );
  std::vector<Point2f> scene_corners(4);

  perspectiveTransform( obj_corners, scene_corners, H);

  Mat homography = img2.clone();
  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  line( homography, scene_corners[0] + Point2f( img1.cols, 0), scene_corners[1] + Point2f( img1.cols, 0), Scalar(0, 255, 0), 4 );
  line( homography, scene_corners[1] + Point2f( img1.cols, 0), scene_corners[2] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );
  line( homography, scene_corners[2] + Point2f( img1.cols, 0), scene_corners[3] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );
  line( homography, scene_corners[3] + Point2f( img1.cols, 0), scene_corners[0] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );


  imshow("homograhy", homography);
}











