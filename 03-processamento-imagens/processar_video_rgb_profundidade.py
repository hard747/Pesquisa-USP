import cv2
import numpy as np
from pathlib import Path

DADOS = Path(__file__).resolve().parent.parent / "dados"

def process_depth_canny(depth_frame):
    return cv2.Canny(depth_frame, 100, 200)

def process_rgb_orb(rgb_frame):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(rgb_frame, None)
    return cv2.drawKeypoints(rgb_frame, kp, None, color=(0,255,0), flags=0)

def main(rgb_video_path, depth_video_path, output_video_path):
    cap_rgb = cv2.VideoCapture(rgb_video_path)
    cap_depth = cv2.VideoCapture(depth_video_path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Cambiado a formato de video común

    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (640*2,480*2))

    while(cap_rgb.isOpened() and cap_depth.isOpened()):
        ret_rgb, frame_rgb = cap_rgb.read()
        ret_depth, frame_depth = cap_depth.read()

        if not ret_rgb or not ret_depth:
            break

        depth_canny = process_depth_canny(frame_depth)
        rgb_orb = process_rgb_orb(frame_rgb)

        depth_canny_color = cv2.cvtColor(depth_canny, cv2.COLOR_GRAY2BGR)

        top_row = np.hstack((frame_rgb, frame_depth))
        bottom_row = np.hstack((rgb_orb, depth_canny_color))
        final_frame = np.vstack((top_row, bottom_row))

        out.write(final_frame)

    cap_rgb.release()
    cap_depth.release()
    out.release()

if __name__ == '__main__':
    main(str(DADOS / 'videos' / 'video_rgb.mp4'), str(DADOS / 'videos' / 'video_cinza.mp4'), 'output_video.avi')  # Cambiado a formato de video común
