#ifndef WRISTDETECT_H
#define WRISTDETECT_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <librealsense2/rs.hpp>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <IrisLandmark.hpp>



namespace MM{

namespace nose{

    class noseDetect
    {
        public:
            cv::Mat _raw_color_image;
            cv::Mat _raw_depth_image;
            cv::Mat _aligned_depth_image;
            cv::Mat _detected_image;
        
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointcloud;

            pcl::PointXYZRGB _search_point;    // watch center point in pcl
            cv::Point2f _nose_center_point;    //watch center point in image

            Eigen::Affine3f _affine;
            Eigen::Matrix3f rotation;
            Eigen::Vector4f _normal;
            int continue_flag;


            
        public:
            noseDetect():_pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>),config(0){}

            noseDetect(cv::Mat& color_image):_raw_color_image(color_image),
                        _pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>){}

            ~noseDetect();

            // Set Image
            void setImage(cv::Mat& color_image, cv::Mat& depth_iamge);
            // Set Point cloud
            void setPointCloud(rs2::video_frame&, rs2::depth_frame&);
            // Get wrist point
            void getnosePoint(my::IrisLandmark& irisLandmarker, rs2_intrinsics& intr);
            //Get wrist pose
            void getnosePose(rs2::depth_frame&);
 
            
            // Show kinds of image
            void showDetectedImage() const;
            
            void showDetectedPointCloud();

        public:

            rs2_intrinsics _intrin;
            int config;

            double getDistance(cv::Point2f point1, cv::Point2f point2);
            pcl::PointXYZ getProjectedPoint(pcl::PointXYZ&, Eigen::Vector4f&);
            
            void getAffine(Eigen::Vector3f&, pcl::PointXYZ&);

            static void matrix2quaternion(Eigen::Matrix4d&, double*);
            static void quaternion2matrix(Eigen::Matrix4d&, double*);
    };


};
};

#endif