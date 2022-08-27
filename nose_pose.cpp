#include <noseDetect.h>
#include <IrisLandmark.hpp>



int main(int argc,char** argv)
{

    my::IrisLandmark irisLandmarker("/home/yxw/Project/nose_pose_detection/nose_pose/models");
    std::cout << "test" << std::endl;
    rs2::pipeline pipe;
    rs2::config cfg;
    std::cout << "test" << std::endl;
    cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 30);
    auto config = pipe.start(cfg);

    auto profile = config.get_stream(RS2_STREAM_COLOR);
    auto intr = profile.as<rs2::video_stream_profile>().get_intrinsics();
    std::cout << "test" << std::endl;
    while(1)
    {
        
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::align align_to_color(RS2_STREAM_COLOR);
        rs2::frameset alignedframe = align_to_color.process(frames);

        auto depth = alignedframe.get_depth_frame();
        auto color = alignedframe.get_color_frame();

        const int w_c = color.as<rs2::video_frame>().get_width();
        const int h_c = color.as<rs2::video_frame>().get_height();

        cv::Mat image_color(cv::Size(w_c, h_c), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat image_depth(cv::Size(w_c, h_c), CV_16U, (void*)depth.get_data(), cv::Mat::AUTO_STEP);

        MM::nose::noseDetect dec1;

        dec1.setImage(image_color, image_depth);
        dec1.getnosePoint(irisLandmarker, intr);
        dec1.showDetectedImage();
        if (cv::waitKey(1) == 112) // low case 'p' key
        {
            dec1.setPointCloud(color, depth);
            dec1.getnosePose(depth);
            dec1.showDetectedPointCloud();
        
            break;
        }
    }
    return 0;
}