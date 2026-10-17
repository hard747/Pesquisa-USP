#include <librealsense2/rs.hpp> // Incluye la API de RealSense
#include <opencv2/opencv.hpp>   // Incluye OpenCV

int main() {
    // Inicializa el contexto de RealSense
    rs2::context ctx;
    rs2::config cfg;

    // Habilita el flujo de color
    cfg.enable_stream(RS2_STREAM_COLOR);

    // Inicia el pipeline
    rs2::pipeline pipe(ctx);
    rs2::pipeline_profile selection = pipe.start(cfg);

    // Captura un fotograma
    rs2::frameset frames = pipe.wait_for_frames();

    // Obtiene el fotograma de color
    rs2::frame color_frame = frames.get_color_frame();

    // Convierte el fotograma a una matriz OpenCV
    cv::Mat color_image(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

    // Muestra la imagen usando OpenCV
    cv::imshow("RealSense SR300", color_image);
    cv::waitKey(0); // Espera hasta que se presione una tecla

    return 0;
}
