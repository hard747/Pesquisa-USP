#include <libfreenect/libfreenect.hpp>
#include <iostream>
#include <fstream>

int main() {
    // Inicializar el contexto de Kinect
    freenect_context *f_ctx;
    if (freenect_init(&f_ctx, nullptr) < 0) {
        std::cerr << "Error al inicializar el contexto de Kinect." << std::endl;
        return 1;
    }

    // Abrir el primer dispositivo
    freenect_device *f_dev;
    if (freenect_open_device(f_ctx, &f_dev, 0) < 0) {
        std::cerr << "No se pudo abrir el dispositivo Kinect." << std::endl;
        return 1;
    }

    // Iniciar streams RGB y Depth
    freenect_set_led(f_dev, LED_RED); // Encender la luz roja
    freenect_set_video_mode(f_dev, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB));
    freenect_set_depth_mode(f_dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED));

    // Crear archivos para guardar las imágenes
    std::ofstream rgbFile("rgb_image.raw", std::ios::binary);
    std::ofstream depthFile("depth_image.raw", std::ios::binary);

    // Buffer para las imágenes
    uint8_t *rgbBuffer = new uint8_t[freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB).bytes];
    uint16_t *depthBuffer = new uint16_t[freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED).bytes / 2];

    // Obtener un cuadro
    freenect_frame_mode video_mode = freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB);
    freenect_frame_mode depth_mode = freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED);

    freenect_set_video_buffer(f_dev, rgbBuffer);
    freenect_set_depth_buffer(f_dev, depthBuffer);

    freenect_start_video(f_dev);
    freenect_start_depth(f_dev);

    // Esperar a que haya datos disponibles
    freenect_process_events(f_ctx);

    // Guardar las imágenes en archivos
    rgbFile.write(reinterpret_cast<const char*>(rgbBuffer), video_mode.bytes);
    depthFile.write(reinterpret_cast<const char*>(depthBuffer), depth_mode.bytes);

    // Limpiar y cerrar el dispositivo
    freenect_stop_depth(f_dev);
    freenect_stop_video(f_dev);
    freenect_close_device(f_dev);
    freenect_shutdown(f_ctx);

    // Liberar memoria
    delete[] rgbBuffer;
    delete[] depthBuffer;

    return 0;
}
