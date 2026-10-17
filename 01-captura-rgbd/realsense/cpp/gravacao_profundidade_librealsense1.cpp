#include <librealsense/rs.hpp>
#include <opencv2/opencv.hpp>

int main() {
    rs::context ctx;
    rs::device* dev = ctx.get_device(0);

    dev->enable_stream(rs::stream::depth, 640, 480, rs::format::z16, 30);

   // cv::VideoWriter video("out.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30, cv::Size(640, 480), false);
    cv::VideoWriter video("out.avi", cv::VideoWriter::fourcc('H', '2', '6', '4'), 30, cv::Size(640, 480), false);
    dev->start();

    for(int i = 0; i < 1000; i++) {
        dev->wait_for_frames();
        const uint16_t* depth_frame = reinterpret_cast<const uint16_t*>(dev->get_frame_data(rs::stream::depth));
        cv::Mat depth_image(cv::Size(640, 480), CV_16U, (void*)depth_frame, cv::Mat::AUTO_STEP);

        // Convert depth image to 8-bit for visualization
        cv::Mat depth_8bit;
        depth_image.convertTo(depth_8bit, CV_8U, 255.0/1000); // Scale factor for visualization

        cv::cvtColor(depth_8bit, depth_8bit, cv::COLOR_GRAY2BGR); // Convert to 3 channels for video writing

        video.write(depth_8bit);
    }

    dev->stop();
    video.release();

    return 0;
}
