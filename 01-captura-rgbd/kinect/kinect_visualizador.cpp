#include <libfreenect/libfreenect.hpp>
#include <opencv2/opencv.hpp>

class KinectViewer : public Freenect::FreenectDevice
{
public:
    KinectViewer(freenect_context *ctx, int index)
        : Freenect::FreenectDevice(ctx, index), rgbImage(FREENECT_VIDEO_RGB), depthImage(FREENECT_DEPTH_11BIT)
    {}

    void VideoCallback(void* _rgb, uint32_t timestamp)
    {
        cv::Mat rgb(480, 640, CV_8UC3, _rgb);
        cv::imshow("RGB", rgb);
        cv::waitKey(1);
    }

    void DepthCallback(void* _depth, uint32_t timestamp)
    {
        cv::Mat depth(480, 640, CV_16UC1, _depth);
        // Puedes procesar el stream de profundidad aquí si es necesario
    }

private:
    cv::Mat rgbImage;
    cv::Mat depthImage;
};

int main()
{
    freenect_context *ctx;
    if (freenect_init(&ctx, nullptr) < 0) {
        std::cerr << "Error al inicializar el contexto de Kinect." << std::endl;
        return 1;
    }

    if (freenect_num_devices(ctx) < 1) {
        std::cerr << "No se encontró ningún dispositivo Kinect." << std::endl;
        return 1;
    }

    freenect_set_log_level(ctx, FREENECT_LOG_INFO);
    freenect_select_subdevices(ctx, static_cast<freenect_device_flags>(FREENECT_DEVICE_CAMERA));

    KinectViewer *viewer = &ctx->devices[0]->getDevice<KinectViewer>();
    viewer->startVideo();
    viewer->startDepth();

    std::cout << "Presiona cualquier tecla para salir." << std::endl;
    cv::waitKey(0);

    viewer->stopVideo();
    viewer->stopDepth();

    freenect_shutdown(ctx);
    return 0;
}
