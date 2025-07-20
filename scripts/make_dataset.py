#!/usr/bin/env python3

import numpy as np
import roslib
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
import copy
import yaml
from std_srvs.srv import SetBool, SetBoolResponse

def simulate_yaw_rotation(image, yaw_angle_degrees, camera_fov):
    """
    ロール成分を抑制した、純粋なヨー方向の回転をシミュレートする関数
    
    Parameters:
    - image: 入力画像
    - yaw_angle_degrees: ヨー方向の回転角度（度）
    - camera_fov: カメラの視野角（度）
    
    Returns:
    - ヨー方向に回転した画像（ロール成分なし）
    """
    height, width = image.shape[:2]
    
    # カメラの内部パラメータ
    f = (width / 2) / np.tan(np.radians(camera_fov / 2))
    K = np.array([
        [f, 0, width/2],
        [0, f, height/2],
        [0, 0, 1]
    ])
    
    # 水平方向のシフトを計算（純粋な水平視点変更）
    yaw_rad = np.radians(yaw_angle_degrees)
    horizontal_shift = f * np.tan(yaw_rad)
    
    # 画像の各ピクセル座標を生成
    y, x = np.indices((height, width))
    
    # 水平シフトのみを適用した新しい座標を計算
    map_x = x + horizontal_shift
    map_y = y.astype(np.float32)
    
    # リマッピングで画像を変換
    warped_image = cv2.remap(image, map_x.astype(np.float32), map_y, 
                          interpolation=cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_CONSTANT)

    return warped_image

def simulate_disparity(img_path, angle_deg=5):
    img = img_path
    h, w = img.shape[:2]
    horizontal_fov = 2.46
    f = (w / 2) / np.tan(horizontal_fov / 2)
    cx, cy = w / 2, h / 2
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])
    angle_rad = np.radians(angle_deg)
    R = np.array([[ np.cos(angle_rad), 0, np.sin(angle_rad)],
                  [0,                 1, 0],
                  [-np.sin(angle_rad),0, np.cos(angle_rad)]])
    H = K @ R @ np.linalg.inv(K)
    H /= H[2, 2]
    warped = cv2.warpPerspective(img, H, (w, h))
    return warped

def preprocess(img):
    h, w = img.shape[:2]
    crop_size = min(h, w)
    left = (w - crop_size) // 2
    img_crop = img[:, left:left+crop_size]
    img_resized = resize(img_crop, (224, 224), mode='constant')
    return img_resized

def shift_image(img, shift_pixels, direction='left'):
    """
    画像を左右にシフトし、空いた部分を黒で塗りつぶす

    Parameters:
        img (np.ndarray): 入力画像（H×W×C または H×W）
        shift_pixels (int): シフトするピクセル数（正の整数）
        direction (str): 'left' または 'right'

    Returns:
        np.ndarray: シフト後の画像
    """
    h, w = img.shape[:2]
    shifted = np.zeros_like(img)

    if shift_pixels >= w:
        return shifted  # 全部黒になる

    if direction == 'right':
        if img.ndim == 2:
            shifted[:, :w - shift_pixels] = img[:, shift_pixels:]
        else:
            shifted[:, :w - shift_pixels, :] = img[:, shift_pixels:, :]
    elif direction == 'left':
        if img.ndim == 2:
            shifted[:, shift_pixels:] = img[:, :w - shift_pixels]
        else:
            shifted[:, shift_pixels:, :] = img[:, :w - shift_pixels, :]
    else:
        raise ValueError("direction must be 'left' or 'right'")

    return shifted

class dataset_creator_node:
    def __init__(self):
        rospy.init_node('dataset_creator_node', anonymous=True)
        self.num = int(rospy.get_param("/dataset_creator_node/num", "1"))
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera_center/image_raw", Image, self.callback)
        self.vel = Twist()
        self.vel_sub = rospy.Subscriber("/cmd_vel", Twist, self.callback_vel)
        self.action = 0.0
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.camera_angle = 10  # 左右カメラの取り付け角度（10度）
        self.camera_fov = 120  # カメラの視野角（適宜調整）
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
        # buttons[1] が押されているかチェック
        if data.buttons[1] == 1:
            self.joy_flg = True

        if data.buttons[6] == 1:
            self.cmd_dir = (0, 1, 0)
        elif data.buttons[7] == 1:
            self.cmd_dir = (0, 0, 1)
        else:
            self.cmd_dir = (1, 0, 0)

        if data.buttons[5] == 1:
            self.inter_flg = True
        else: 
            self.inter_flg = False

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

        if inter_flg == True or cmd_dir != (1, 0, 0):
            inter = (0, 1)
        else:
            inter = (1, 0)

        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['episode', 'intersection'])
            writer.writerow([episode, inter])

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.cmd_dir == (0, 0, 0):
            return
        print(self.episode)

        cv_left_image = simulate_yaw_rotation(self.cv_image, -self.camera_angle, self.camera_fov)   # -10度
        cv_right_image = simulate_yaw_rotation(self.cv_image, self.camera_angle, self.camera_fov) # +10度
        
        # cv_left_image = shift_image(self.cv_image, 60, direction='left')
        # cv_right_image = shift_image(self.cv_image, 60, direction='right')
        img_center = cv2.resize(self.cv_image, (64, 48))
        img_left = cv2.resize(cv_left_image, (64, 48))
        img_right = cv2.resize(cv_right_image, (64, 48))

        # img_center = resize(self.cv_image, (48, 64), mode='constant')
        # img_left = resize(cv_left_image, (48, 64), mode='constant')
        # img_right = resize(cv_right_image, (48, 64), mode='constant')

        # img_center = preprocess(self.cv_image)
        # img_left = preprocess(cv_left_image)
        # img_right = preprocess(cv_right_image)
        self.save_image("center", self.episode, img_center)
        self.save_image("left",   self.episode, img_left)
        self.save_image("right",  self.episode, img_right)

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
        # cv2.imshow("center", self.cv_image)
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