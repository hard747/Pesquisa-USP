import cv2
import subprocess

def capture_images():
    subprocess.run(['guvcview', '--no_display', '--save=rgb_image.jpg'])
    subprocess.run(['guvcview', '--no_display', '--save=depth_image.png'])

def process_depth_canny(depth_image_path):
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    depth_canny = cv2.Canny(depth_image, 100, 200)
    cv2.imwrite('depth_canny.png', depth_canny)

def process_rgb_orb(rgb_image_path):
    rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(rgb_image, None)
    rgb_orb = cv2.drawKeypoints(rgb_image, kp, None, color=(0,255,0), flags=0)
    cv2.imwrite('rgb_orb.jpg', rgb_orb)

def show_processed_images():
    depth_canny = cv2.imread('depth_canny.png', cv2.IMREAD_COLOR)
    rgb_orb = cv2.imread('rgb_orb.jpg', cv2.IMREAD_COLOR)

    cv2.imshow('Depth Image with Canny Edges', depth_canny)
    cv2.imshow('RGB Image with ORB Keypoints', rgb_orb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_images()
    process_depth_canny('depth_image.png')
    process_rgb_orb('rgb_image.jpg')
    show_processed_images()
