#include <ros/ros.h>

#include <iostream>
#include <popl.hpp>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <camera_info_manager/camera_info_manager.h>

int main(int argc, char* argv[]) {
    ros::init(argc, argv, "video_left_publisher");

    // create options
    popl::OptionParser op("Allowed options");
    auto help = op.add<popl::Switch>("h", "help", "produce help message s");
    auto video_file_path = op.add<popl::Value<std::string>>("m", "video", "video file path");
    try {
        op.parse(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // check validness of options
    if (help->is_set()) {
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    if (!video_file_path->is_set()) {
        std::cerr << "invalid arguments" << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // initialize this node
    const ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    const image_transport::Publisher publisher = it.advertise("/video/left/image_raw", 1);
    //ros::Publisher pub_info_left = nh.advertise<sensor_msgs::CameraInfo>("/stereo/left/camera_info", 1);
    //sensor_msgs::CameraInfo info_left;
   // camera_info_manager::CameraInfoManager caminfo(nh, "test-camera",  "file:///home/zy2020/openvslam/ros/camera.yaml");

    ros::NodeHandle nh2;
    const std::string cname="test";
    const std::string url="file:///home/zy2020/openvslam/ros/camera.yaml";
    ros::Publisher pub_info_left= nh2.advertise<sensor_msgs::CameraInfo>("/stereo/left/camera_info", 1);
    camera_info_manager::CameraInfoManager caminfo(nh,cname,url);
    sensor_msgs::CameraInfo info_left;

    cv::Mat frame;
    cv::VideoCapture video;
    sensor_msgs::ImagePtr msg;

    // load video file
    if (!video.open(video_file_path->value(), cv::CAP_FFMPEG)) {
        std::cerr << "can't load video file" << std::endl;
        std::cerr << std::endl;
        return EXIT_FAILURE;
    }

    ros::Rate pub_rate(video.get(cv::CAP_PROP_FPS));

    while (nh.ok() && video.read(frame)) {
        // send message
        msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
        publisher.publish(msg);

        //info_left.width = calib_data["image_width"];
        //info_left.height = calib_data["image_height"];
        //info_left.K = calib_data["camera_matrix"]["data"];
        //info_left.D = calib_data["distortion_coefficients"]["data"];
        //info_left.R = calib_data["rectification_matrix"]["data"];
        //info_left.P = calib_data["projection_matrix"]["data"];
        //info_left.distortion_model = calib_data["distortion_model"] ;
        info_left.header.stamp = ros::Time::now();
        info_left.header.frame_id="map";
        info_left=caminfo.getCameraInfo();
        pub_info_left.publish(info_left);

        ros::spinOnce();
        pub_rate.sleep();
    }
    return EXIT_SUCCESS;
}