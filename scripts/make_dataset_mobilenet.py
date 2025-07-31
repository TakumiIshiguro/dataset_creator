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

def simulate_left_right_crop(image):
    """
    入力画像から、左カメラ・右カメラ視点に相当する画像を生成する。

    - 左カメラ視点: 左端から正方形をクロップ
    - 右カメラ視点: 右端から正方形をクロップ

    Parameters:
        image (np.ndarray): 入力画像（H×W×C）

    Returns:
        left_image (np.ndarray): 左カメラ視点（左端をクロップ）
        right_image (np.ndarray): 右カメラ視点（右端をクロップ）
    """
    h, w = image.shape[:2]
    crop_size = min(h, w)

    # 左カメラ視点 → 左端から正方形
    left_image = image[:, :crop_size]

    # 右カメラ視点 → 右端から正方形
    right_image = image[:, w - crop_size:]

    return left_image, right_image

def simulate_left_right_disparity_from_crop(image, angle_deg=5, crop_ratio=0.9):
    """
    左右端をクロップし、それぞれに射影変換を適用する。
    - 左画像は右に回転（右向き）
    - 右画像は左に回転（左向き）
    → これにより仮想的な視差の方向が実際のステレオカメラに近くなる

    Parameters:
        image (np.ndarray): 入力画像（H×W×C）
        angle_deg (float): 射影変換角度
        crop_ratio (float): クロップする画像の割合

    Returns:
        left_img (np.ndarray), right_img (np.ndarray)
    """
    h, w = image.shape[:2]
    crop_size = int(min(h, w) * crop_ratio)
    cx, cy = crop_size / 2, h / 2

    left_crop  = image[:, :crop_size]
    right_crop = image[:, w - crop_size:]

    def warp_view(img, angle_deg):
        horizontal_fov = 2.35
        f = (crop_size / 1.5) / math.tan(horizontal_fov / 2)
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]])
        angle_rad = math.radians(angle_deg)
        R = np.array([[ math.cos(angle_rad), 0, math.sin(angle_rad)],
                      [0, 1, 0],
                      [-math.sin(angle_rad), 0, math.cos(angle_rad)]])
        H = K @ R @ np.linalg.inv(K)
        H /= H[2, 2]
        return cv2.warpPerspective(img, H, (crop_size, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)

    # ✨ 回転方向を左右反転 ✨
    left_img  = warp_view(left_crop,  +angle_deg)  # 右向き
    right_img = warp_view(right_crop, -angle_deg)  # 左向き

    return left_img, right_img

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
        self.image_sub = rospy.Subscriber("/camera_center/image_raw", Image, self.callback)
        self.vel = Twist()
        self.vel_sub = rospy.Subscriber("/cmd_vel", Twist, self.callback_vel)
        self.action = 0.0
        # self.cv_image = np.zeros((480,640,3), np.uint8)
        self.cv_image = np.zeros((720, 1280, 3), np.uint8)
        self.cmd_dir = (0, 0, 0)
        self.episode = 1
        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)
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
        if data.buttons[1] == 1:
            self.joy_flg = True
        if data.buttons[6] == 1:
            self.cmd_dir = (0, 1, 0)
        elif data.buttons[7] == 1:
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
