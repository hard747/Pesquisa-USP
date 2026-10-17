#include <librealsense2/rs.hpp> 
#include <opencv2/opencv.hpp>

int main() {
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

    pipe.start(cfg);

    cv::namedWindow("RealSense Display", cv::WINDOW_AUTOSIZE);

    while (true) {
        rs2::frameset frames = pipe.wait_for_frames();

        rs2::frame color_frame = frames.get_color_frame();
        const int w = color_frame.as<rs2::video_frame>().get_width();
        const int h = color_frame.as<rs2::video_frame>().get_height();

        cv::Mat image(cv::Size(w, h), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

        cv::imshow("RealSense Display", image);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    pipe.stop();
    cv::destroyAllWindows();

    return 0;
}

