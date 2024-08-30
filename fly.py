import cv2
import math
import random
import numpy as np
from time import time

class fly:
    def __init__(self, x, y, speed, size = 5, max_size = 6, sumin = 0, skin="assets/fly.png"):
        self.speed = speed
        self.x = x
        self.y = y
        self.size = size
        self.x2 = 0
        self.y2 = 0
        self.end_x = 0
        self.end_y = 0

        self.point = self.speed * abs(max_size - self.size)

        # 작으면 작을수록 더 높은 점수를

        self.degree = 0
        if sumin == 0 or sumin == 1:
            self.skin = cv2.imread("assets/fly.png", cv2.IMREAD_UNCHANGED)  # 알파 채널 포함하여 이미지 읽기
        elif sumin == 2:
            self.skin = cv2.imread(skin, cv2.IMREAD_UNCHANGED)  # 알파 채널 포함하여 이미지 읽기

        if sumin == 1:
            self.sumin()
        # self.x_end = self.skin.size()
        # self.y_end = self.ski
        if self.skin is None:
            raise FileNotFoundError("fly.png 파일을 찾을 수 없습니다.")
        self.appear(None, x, y)

    def resize_image(self, image, scale_percent):
        # 원본 이미지의 크기를 가져옵니다.
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        # 크기를 조정합니다.
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        
        return resized

    def cal_deg(self, x1, y1, x2, y2):
        # 각도 계산 (라디안 단위)
        angle_rad = math.atan2(y2 - y1, x2 - x1)
        # 각도를 도 단위로 변환
        self.degree = math.degrees(angle_rad)
        self.degree += 90

    def overlay_image(self, background, overlay, x, y):
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        
        h, w, _ = overlay.shape

        # Rotate the overlay image
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.degree, 1.0)
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        rotated_overlay = cv2.warpAffine(overlay, rotation_matrix, (new_w, new_h))

        new_h, new_w, _ = rotated_overlay.shape

        # Ensure the ROI is within the background image bounds
        if x + new_w > background.shape[1]:
            new_w = background.shape[1] - x
        if y + new_h > background.shape[0]:
            new_h = background.shape[0] - y

        # If new_w or new_h becomes 0 after the adjustments, return the background as is
        if new_w <= 0 or new_h <= 0:
            return background

        # Resize the overlay to match the ROI dimensions if they don't match
        roi = background[y:y+new_h, x:x+new_w]

        # Resize the rotated_overlay to match the ROI dimensions
        resized_overlay = cv2.resize(rotated_overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

        overlay_img = resized_overlay[:, :, :3]
        mask = resized_overlay[:, :, 3]

        if mask.shape[1] == 0 or mask.shape[0] == 0:
            return background

        alpha = cv2.merge([mask, mask, mask]) / 255.0
        alpha_inv = 1.0 - alpha

        # print("x:", x)
        # print("y:", y)
        # print("roi shape:", roi.shape)
        # print("overlay_img shape:", overlay_img.shape)
        # print("alpha shape:", alpha.shape)
        # print("alpha_inv shape:", alpha_inv.shape)

        for c in range(0, 3):
            roi[:, :, c] = (alpha_inv[:, :, c] * roi[:, :, c] + alpha[:, :, c] * overlay_img[:, :, c])

        background[y:y+new_h, x:x+new_w] = roi

        return background
       
    def appear(self, img, x, y):
        if img is None:
            return None
        self.end_cor()
        return self.overlay_image(img, self.resize_image(self.skin, self.size), x, y)

    def updateXY(self, x, y):
        if self.x < x:
            self.x += self.speed
        elif self.x > x:
            self.x -= self.speed
        if self.y < y:
            self.y += self.speed
        elif self.y > y:
            self.y -= self.speed

    def move_straight(self, img, target_x, target_y):
        if abs(self.x2 - self.x) < self.speed and abs(self.y2 - self.y) < self.speed:
            self.x2 = target_x
            self.y2 = target_y
        self.updateXY(self.x2, self.y2)
        self.cal_deg(self.x, self.y, self.x2, self.y2)
        return self.appear(img, self.x, self.y)
    
    def end_cor(self):
        resize = self.resize_image(self.skin, self.size)
        self.end_x = self.x + resize.shape[0]
        self.end_y = self.y + resize.shape[1]

    def add_alpha_channel(self, image):
        # 이미 알파 채널이 있으면 그대로 반환
        if image.shape[2] == 4:
            return image
        # 알파 채널이 없으면 알파 채널 추가
        b, g, r = cv2.split(image)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255  # 모든 픽셀에 대해 알파 채널 값을 255로 설정 (불투명)
        image_with_alpha = cv2.merge([b, g, r, alpha])
        return image_with_alpha 


    def sumin(self):
        sumin_img = cv2.imread("assets/sumin1.png", cv2.IMREAD_UNCHANGED)
        if sumin_img is None:
            raise FileNotFoundError("sumin.jpg 파일을 찾을 수 없습니다.")
        
        sumin_img_with_alpha = self.add_alpha_channel(sumin_img)  # 알파 채널이 없는 경우 추가
        self.skin = self.overlay_image(
            self.skin,
            self.resize_image(sumin_img_with_alpha, 35),
            210,
            0
        )
