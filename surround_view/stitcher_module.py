"""
此模块用于将四个视角的图像拼接成一个鸟瞰图。
"""
import os
import cv2
import numpy as np
from surround_view import FisheyeCameraModel, BirdView
import surround_view.param_settings as settings
import concurrent.futures

class BirdViewStitcher:
    def __init__(self, init_images=None):
        # === 加载标定和相机模型 ===
        names = settings.camera_names
        yamls = [os.path.join(os.getcwd(), "yaml", name + ".yaml") for name in names]
        self.camera_models = [FisheyeCameraModel(camera_file, camera_name) 
                              for camera_file, camera_name in zip(yamls, names)]

        # === 初始化 birdview 模块 ===
        self.birdview = BirdView()

        # === 加载静态的权重图和融合掩码，只做一次 ===
        self.birdview.load_weights_and_masks("./weights.png", "./masks.png")
        if init_images is not None:
            processed = [self.process_frame(img, cam) for img, cam in zip(init_images, self.camera_models)]
            self.birdview.get_weights_and_masks(processed)

       

    def _dummy_image(self):
        return (255 * np.ones((720, 1280, 3), dtype=np.uint8))  # 改为你相机的输入分辨率

    def process_frame(self, frame, camera):
        img = camera.undistort(frame)
        img = camera.project(img)
        img = camera.flip(img)
        return img

    def stitch_frames(self, front_frame, back_frame, left_frame, right_frame):
        # 多线程处理图像
        frames = [front_frame, back_frame, left_frame, right_frame]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            processed_frames = list(executor.map(self.process_frame, frames, self.camera_models))

        # ✅ 不再重复计算 get_weights_and_masks
        self.birdview.update_frames(processed_frames)
        self.birdview.make_luminance_balance().stitch_all_parts()
        self.birdview.make_white_balance()
        self.birdview.copy_car_image()
        return self.birdview.image

    def __del__(self):
        """析构函数，关闭多线程执行器"""
        self.executor.shutdown()