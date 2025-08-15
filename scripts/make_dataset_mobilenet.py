#!/usr/bin/env python3

import numpy as np
import roslib
import math
roslib.load_manifest('dataset_creator')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from skimage.transform import resize
import os
import sys
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import csv
import time
from sensor_msgs.msg import Joy
from std_srvs.srv import SetBool, SetBoolResponse

def simulate_left_right_disparity_from_crop(image, angle_deg=5, fov_deg=90):
    """
    射影変換・リサイズを行わず、中央画像から左右視点画像を端からクロップして生成。

    Parameters:
    - image: 入力画像（H×W×3）
    - angle_deg: 無視（互換性のために残している）
    - fov_deg: 無視（互換性のために残している）

    Returns:
    - left_img: 左視点画像（画像左端から切り出し）
    - right_img: 右視点画像（画像右端から切り出し）
    """
    h, w = image.shape[:2]
    crop_width = int(w * 0.8)  # 中央画像と被らない程度にクロップ幅を設定

    # 左端からクロップ
    left_img = image[:, 0:crop_width]

    # 右端からクロップ
    right_img = image[:, w - crop_width:w]

    return left_img, right_img

# def simulate_left_right_disparity_from_crop(image, angle_deg=5, fov_deg=130):
#     """
#     射影変換を使って左右視点画像を生成する。
#     入力画像の左右端を切り出し、それぞれに視点の回転（ヨー方向）を加える。

#     Parameters:
#     - image: 入力画像（H×W×3）
#     - angle_deg: 左右の視点ずらし角度（度）
#     - fov_deg: 水平方向の視野角（度）

#     Returns:
#     - left_img: 左視点画像
#     - right_img: 右視点画像
#     """
#     h, w = image.shape[:2]
#     crop_width = int(w * 0.8)

#     # === 左右クロップ ===
#     left_crop = image[:, 0:crop_width]
#     right_crop = image[:, w - crop_width:w]

#     # === カメラ内部行列Kを定義 ===
#     fov_rad = math.radians(fov_deg)
#     f = 0.5 * crop_width / math.tan(fov_rad / 2)
#     cx = crop_width / 2
#     cy = h / 2
#     K = np.array([
#         [f, 0, cx],
#         [0, f, cy],
#         [0, 0, 1]
#     ])
#     K_inv = np.linalg.inv(K)

#     # === 射影変換用回転行列（ヨー回転） ===
#     def get_yaw_homography(angle_rad):
#         R = np.array([
#             [math.cos(angle_rad), 0, math.sin(angle_rad)],
#             [0, 1, 0],
#             [-math.sin(angle_rad), 0, math.cos(angle_rad)]
#         ])
#         return K @ R @ K_inv

#     angle_rad = math.radians(angle_deg)
#     H_left = get_yaw_homography(-angle_rad)
#     H_right = get_yaw_homography(angle_rad)

#     # === warpPerspectiveで射影変換を適用 ===
#     left_img = cv2.warpPerspective(left_crop, H_left, (crop_width, h), flags=cv2.INTER_LINEAR)
#     right_img = cv2.warpPerspective(right_crop, H_right, (crop_width, h), flags=cv2.INTER_LINEAR)

#     return left_img, right_img

def preprocess_for_mobilenet(img):
    """
    MobileNetV3 用に画像を正方形にクロップし、224×224にリサイズしてuint8で返す
    """
    h, w = img.shape[:2]
    crop_size = min(h, w)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    img_crop = img[top:top+crop_size, left:left+crop_size]
    img_resized = resize(img_crop, (224, 224), mode='constant')
    return (img_resized * 255).astype(np.uint8)

class dataset_creator_node:
    def __init__(self):
        rospy.init_node('dataset_creator_node', anonymous=True)
        self.num = int(rospy.get_param("/dataset_creator_node/num", "1"))
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera_center/image_raw", Image, self.callback, queue_size=1)
        self.vel = Twist()
        self.vel_sub = rospy.Subscriber("/cmd_vel", Twist, self.callback_vel, queue_size=1)
        self.action = 0.0
        # self.cv_image = np.zeros((480,640,3), np.uint8)
        self.cv_image = np.zeros((720, 1280, 3), np.uint8)
        self.cmd_dir = (0, 0, 0)
        self.episode = 1
        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback, queue_size=1)
        self.joy_flg = False
        self.inter_flg = False
        self.loop_srv = rospy.Service('/loop_count', SetBool, self.callback_loop_count)
        self.loop_count_flag = False
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.save_path = roslib.packages.get_pkg_dir('dataset_creator') + '/dataset/' + str(self.start_time)

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel.angular.z

    def callback_loop_count(self, data):
        resp = SetBoolResponse()
        self.loop_count_flag = data.data
        resp.success = True
        return resp

    def joy_callback(self, data):
        # if data.buttons[1] == 1:
        #     self.joy_flg = True
        # if data.buttons[6] == 1:
        if data.axes[2] < 0:
            self.cmd_dir = (0, 1, 0)
        # elif data.buttons[7] == 1:
        elif data.axes[5] < 0:
            self.cmd_dir = (0, 0, 1)
        else:
            self.cmd_dir = (1, 0, 0)
        self.inter_flg = data.buttons[5] == 1

    def save_image(self, subdir, episode, img):
        img_path = os.path.join(self.save_path, 'image', subdir)
        os.makedirs(img_path, exist_ok=True)
        cv2.imwrite(os.path.join(img_path, f"{episode}.png"), img)

    def save_dir_csv(self, path, episode, cmd_dir):
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, 'dir.csv')
        write_header = not os.path.exists(file_path)
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['episode', 'cmd_dir']) 
            writer.writerow([episode, cmd_dir])

    def save_vel_csv(self, view, episode, angular_z):
        vel_path = os.path.join(self.save_path, 'vel')
        os.makedirs(vel_path, exist_ok=True)
        file_map = {
            "center": "center.csv",
            "left": "left.csv",
            "right": "right.csv"
        }
        filename = file_map[view]
        file_path = os.path.join(vel_path, filename)
        write_header = not os.path.exists(file_path)
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['episode', 'angular_z'])
            writer.writerow([episode, angular_z])

    def save_inter_csv(self, path, episode, cmd_dir, inter_flg):
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, 'inter.csv')
        write_header = not os.path.exists(file_path)
        inter = (0, 1) if inter_flg or cmd_dir != (1, 0, 0) else (1, 0)
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['episode', 'intersection'])
            writer.writerow([episode, inter])

    def loop(self):
        # if self.cv_image.size != 640 * 480 * 3:
        if self.cv_image.size != 1280 * 720 * 3:
            return
        if self.cmd_dir == (0, 0, 0):
            return
        print(self.episode)

        cv_left_image, cv_right_image = simulate_left_right_disparity_from_crop(self.cv_image, angle_deg=5)

        # 画像をMobileNetV3用に前処理
        img_center_uint8 = preprocess_for_mobilenet(self.cv_image)
        img_left_uint8   = preprocess_for_mobilenet(cv_left_image)
        img_right_uint8  = preprocess_for_mobilenet(cv_right_image)

        img_resize = resize(self.cv_image, (224, 224), mode='constant') 
        img_all_resize = (img_resize * 255).astype(np.uint8)

        # 保存
        self.save_image("center", self.episode, img_center_uint8)
        self.save_image("left",   self.episode, img_left_uint8)
        self.save_image("right",  self.episode, img_right_uint8)
        self.save_image("resize", self.episode, img_all_resize)

        views = [
            ("center", self.action),
            ("left", self.action - 0.2),
            ("right", self.action + 0.2)
        ]
        for view, angle in views:
            self.save_vel_csv(view, self.episode, angle)

        self.save_dir_csv(self.save_path, self.episode, self.cmd_dir)
        self.save_inter_csv(self.save_path, self.episode, self.cmd_dir, self.inter_flg)  

        if self.loop_count_flag:
            self.loop_count_flag = False
            os.system('killall roslaunch')
            sys.exit()

        cv2.imshow("left", cv_left_image)
        cv2.imshow("right", cv_right_image)
        cv2.waitKey(1)
        self.episode += 1

if __name__ == '__main__':
    rg = dataset_creator_node()
    DURATION = 0.1
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
