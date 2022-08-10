#include <noseDetect.h>
#include <rs2pcd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <IrisLandmark.hpp>

#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;
using namespace cv::face;


void MM::nose::noseDetect::setImage(cv::Mat& color_image, cv::Mat& depth_image) 
{
    _raw_color_image = color_image;
    _raw_depth_image = depth_image;
    continue_flag = 0;
}

void MM::nose::noseDetect::getnosePoint(my::IrisLandmark& irisLandmarker, rs2_intrinsics& intr)
{
    _detected_image = _raw_color_image.clone();
    std::vector<cv::Point3d> model_points;
    model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // Nose tip
    model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));       // Left eye left corner
    model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));      // Left Mouth corner
    model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));          // Chin
    model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));        // Right eye right corner
    model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));       // Right mouth corner    

    irisLandmarker.loadImageToInput(_detected_image);
    irisLandmarker.runInference();          
    int i = 0;
    std::vector<cv::Point2d> face_model_points;
    std::vector<cv::Point2d> nose_model_points;
    for (auto landmark: irisLandmarker.getAllFaceLandmarks()) {
        cv::circle(_detected_image, landmark, 2, cv::Scalar(0, 255, 0), -1);
        if ( i == 1 || i == 152 || i == 61 || i == 291 || i == 33 || i == 263)
        {
            face_model_points.push_back(landmark);
            cv::circle(_detected_image, landmark, 5, cv::Scalar(0, 255, 0), -1);
        }
        if ( i == 250 || i == 289 || i == 290 || i == 305 || i == 309 || i == 392)
        {
            nose_model_points.push_back(landmark);
            cv::circle(_detected_image, landmark, 3, cv::Scalar(0, 0, 255), -1);
        }
        i++;
    }
    if((face_model_points.size() != 6) && (nose_model_points.size() != 6) )
    {
        continue_flag = 1;
    } else
    {

        cv::Mat rotation_vector; // Rotation in axis-angle form
        cv::Mat camera_matrix  = (cv::Mat_<double>(3,3) << intr.fx, 0, intr.ppx, 0 , intr.fy, intr.ppy, 0, 0, 1);
        cv::Mat translation_vector;

        cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type);

        cv::solvePnP(model_points, face_model_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

        std::vector<cv::Point3d> nose_end_point3D;
        std::vector<cv::Point2d> nose_end_point2D;
        std::vector<cv::Point3d> nose_X_point3D;
        std::vector<cv::Point2d> nose_X_point2D;
        std::vector<cv::Point3d> nose_Y_point3D;
        std::vector<cv::Point2d> nose_Y_point2D;

        nose_end_point3D.push_back(cv::Point3d(0,0,1000.0));
        nose_X_point3D.push_back(cv::Point3d(0,1000.0,0.0));
        nose_Y_point3D.push_back(cv::Point3d(1000.0,0,0));
        projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);
        projectPoints(nose_X_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_X_point2D);
        projectPoints(nose_Y_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_Y_point2D);
        cv::line(_detected_image,face_model_points[0], nose_end_point2D[0], cv::Scalar(255,0,0), 2);
        cv::line(_detected_image,face_model_points[0], nose_X_point2D[0], cv::Scalar(255,0,0), 2);
        cv::line(_detected_image,face_model_points[0], nose_Y_point2D[0], cv::Scalar(255,0,0), 2);


        cv::Point2d nose_center_add(0,0);
        for(int i = 0; i < 6; i++)
        {
            nose_center_add.x += nose_model_points[i].x; 
            nose_center_add.y += nose_model_points[i].y; 
        }
        _nose_center_point.x = (nose_center_add.x / 6.0);
        _nose_center_point.y = (nose_center_add.y / 6.0);

        cv::cv2eigen(rotation_vector, rotation);
    }
    

    
    std::cout << "get nose center Point: " <<  _nose_center_point << std::endl;
}
    




void MM::nose::noseDetect::getnosePose(rs2::depth_frame& depth)
{  
 

    // distance at z axis from camera to watch_center_point
    auto dist = depth.get_distance(_nose_center_point.x, _nose_center_point.y);

    float pointxyz[3];
    float pix[2] = {_nose_center_point.x, _nose_center_point.y};
    // Get the coordinate of watch_center_point in xyz frame
    rs2_deproject_pixel_to_point(pointxyz, &_intrin, pix, dist);
    
    // Get the search point in point cloud
    _search_point.x = pointxyz[0];
    _search_point.y = pointxyz[1];
    _search_point.z = pointxyz[2];

    
}



void MM::nose::noseDetect::showDetectedImage() const
{
    imshow("Facial Landmark Detection", _detected_image);
    
    // cv::waitKey(0);
}


void MM::nose::noseDetect::showDetectedPointCloud()
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("frame show"));

    // Set background of viewer to black
    viewer->setBackgroundColor (0, 0, 0); 
    // Add generated point cloud and identify with string "Cloud"
    viewer->addPointCloud<pcl::PointXYZRGB> (_pointcloud, "frame show");
    // Default size for rendered points
    viewer->setPointCloudRenderingProperties (
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "frame show");
        
    Eigen::Translation3f translation(_search_point.x, _search_point.y, _search_point.z);
    
    _affine = translation * rotation;

    
    viewer->addCoordinateSystem(0.2, _affine);

    
    // viewer->addCoordinateSystem();
    // Viewer Properties
    viewer->initCameraParameters();  // Camera Parameters for ease of viewing

    cout << "control + c to exit the program. " << endl;
    viewer->spin(); 
}

double MM::nose::noseDetect::getDistance(cv::Point2f point1, cv::Point2f point2)
{
    double distance = sqrtf(powf((point1.x - point2.x), 2) 
                        + powf((point1.y - point2.y), 2));
    return distance;
}

void MM::nose::noseDetect::setPointCloud(
    rs2::video_frame& RGB, rs2::depth_frame& depth)
{
    // get point cloud from realsense
    _pointcloud = ::rs2pcd(RGB, depth);

    // get intrinsic of depth frame
    auto color_stream = RGB.get_profile().as<rs2::video_stream_profile>();
    _intrin = color_stream.get_intrinsics();
}


pcl::PointXYZ MM::nose::noseDetect::getProjectedPoint(pcl::PointXYZ& xAxisPointOutPlane, Eigen::Vector4f& planeCoe)
{
    pcl::PointXYZ xAisPointInPlane;
    double A = planeCoe[0];
    double B = planeCoe[1];
    double C = planeCoe[2];
    double D = planeCoe[3];
    double abd_2 = planeCoe[0] * planeCoe[0] + planeCoe[1] * planeCoe[1] + planeCoe[2] * planeCoe[2];

    xAisPointInPlane.x = 
    ((B*B+C*C)*xAxisPointOutPlane.x - A*(B*xAxisPointOutPlane.y + C*xAxisPointOutPlane.z + D))/abd_2;
    xAisPointInPlane.y = 
    ((A*A +C*C)*xAxisPointOutPlane.y -B*(A*xAxisPointOutPlane.x+C*xAxisPointOutPlane.z +D))/abd_2;
    xAisPointInPlane.z = 
    ((A*A+B*B)*xAxisPointOutPlane.z - C*(A*xAxisPointOutPlane.x + B*xAxisPointOutPlane.y +D)) /abd_2;

    return xAisPointInPlane;
}

void MM::nose::noseDetect::getAffine(Eigen::Vector3f& nx_input, pcl::PointXYZ& origin)
{
    Eigen::Vector3f nx, ny, nz, t;
    nx = nx_input;
    nz = {-_normal[0], -_normal[1], -_normal[2]};
    ny = nz.cross(nx);
    nx = ny.cross(nz);
    double nx_mode = std::sqrt(nx[0]*nx[0] + nx[1]*nx[1] + nx[2]*nx[2]);
    double ny_mode = std::sqrt(ny[0]*ny[0] + ny[1]*ny[1] + ny[2]*ny[2]);
    double nz_mode = std::sqrt(nz[0]*nz[0] + nz[1]*nz[1] + nz[2]*nz[2]);

    Eigen::Vector4d nx_homo, ny_homo, nz_homo, t_homo;
    nx_homo = {nx[0]/nx_mode, nx[1]/nx_mode, nx[2]/nx_mode, 0};
    ny_homo = {ny[0]/ny_mode, ny[1]/ny_mode, ny[2]/ny_mode, 0};
    nz_homo = {nz[0]/nz_mode, nz[1]/nz_mode, nz[2]/nz_mode, 0};
    t_homo = {origin.x, origin.y, origin.z, 1};
    
    Eigen::Matrix4d tran;

    tran(0, 0) = nx_homo[0];  
    tran(0, 1) = ny_homo[0];
    tran(0, 2) = nz_homo[0];
    tran(0, 3) = t_homo[0];

    tran(1, 0) = nx_homo[1];
    tran(1, 1) = ny_homo[1];
    tran(1, 2) = nz_homo[1];
    tran(1, 3) = t_homo[1];
    
    tran(2, 0) = nx_homo[2];
    tran(2, 1) = ny_homo[2];
    tran(2, 2) = nz_homo[2];
    tran(2, 3) = t_homo[2];

    tran(3, 0) = nx_homo[3];
    tran(3, 1) = ny_homo[3];
    tran(3, 2) = nz_homo[3];
    tran(3, 3) = t_homo[3];

    Eigen::Matrix4f tranf= tran.cast<float>();


    _affine = tranf;
    std::cout << "The watch wearing point affine matrix: " << '\n' << _affine.matrix() << std::endl;
}

void MM::nose::noseDetect::matrix2quaternion(Eigen::Matrix4d& tran, double* quat)
{
    Eigen::Matrix3d rotation_matrix;
    rotation_matrix = tran.block(0,0,3,3);

    Eigen::Quaterniond quaternion;
    quaternion=rotation_matrix;
    
    quat[0] = tran(0,3);
    quat[1] = tran(0,3);
    quat[2] = tran(0,3);
    quat[3] = quaternion.x();
    quat[4] = quaternion.y();
    quat[5] = quaternion.z();
    quat[6] = quaternion.w();
}

void MM::nose::noseDetect::quaternion2matrix(Eigen::Matrix4d& tran, double* quat)
{

    Eigen::Matrix3d rotation_matrix;

    Eigen::Quaterniond quaternion;
    quaternion.x() = quat[3];
    quaternion.x() = quat[4];
    quaternion.x() = quat[5];
    quaternion.x() = quat[6];

    rotation_matrix = quaternion.matrix();
    
    tran.block(0,0,3,3) = rotation_matrix;

    tran(0,3) = quat[0];
    tran(1,3) = quat[1];
    tran(2,3) = quat[2];
    tran(3,0) = 0;
    tran(3,1) = 0;
    tran(3,2) = 0;
    tran(3,3) = 1;
}

MM::nose::noseDetect::~noseDetect()
{

}
