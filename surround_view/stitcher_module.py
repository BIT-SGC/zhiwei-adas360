# """
# 此模块用于将四个视角的图像拼接成一个鸟瞰图。
# """
# import os
# import cv2
# import numpy as np
# from surround_view import FisheyeCameraModel, BirdView
# import surround_view.param_settings as settings
# import concurrent.futures

# class BirdViewStitcher:
#     """
#     鸟瞰图拼接器类，用于将四个视角的图像拼接成一个鸟瞰图。
#     """
#     def __init__(self):
#         """
#         初始化鸟瞰图拼接器，加载相机模型和权重掩码。
#         """
#         names = settings.camera_names
#         yamls = [os.path.join(os.getcwd(), "yaml", name + ".yaml") for name in names]
#         self.camera_models = [FisheyeCameraModel(camera_file, camera_name) 
#                               for camera_file, camera_name in zip(yamls, names)]
#         self.birdview = BirdView()
#         self.birdview.load_weights_and_masks("./weights.png", "./masks.png")

#     def process_frame(self, frame, camera):
#         img = camera.undistort(frame)
#         img = camera.project(img)
#         img = camera.flip(img)
#         return img

#     def stitch_frames(self, front_frame, back_frame, left_frame, right_frame):
#         """
#         将四个视角的图像拼接成一个鸟瞰图。

#         参数:
#         front_frame (np.ndarray): 前视角图像
#         back_frame (np.ndarray): 后视角图像
#         left_frame (np.ndarray): 左视角图像
#         right_frame (np.ndarray): 右视角图像

#         返回:
#         np.ndarray: 拼接好的鸟瞰图
#         """
#         frames = [front_frame, back_frame, left_frame, right_frame]
        
#         # 使用多线程处理图像
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             processed_frames = list(executor.map(self.process_frame, frames, self.camera_models))

#         self.birdview.get_weights_and_masks(processed_frames)
#         self.birdview.update_frames(processed_frames)
#         self.birdview.make_luminance_balance().stitch_all_parts()
#         self.birdview.make_white_balance()
#         self.birdview.copy_car_image()
#         return self.birdview.image


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

    def process_frame(self, frame, camera):
        img = camera.undistort(frame)
        img = camera.project(img)
        img = camera.flip(img)
        return img

    def stitch_frames(self, front_frame, back_frame, left_frame, right_frame):
        frames = [front_frame, back_frame, left_frame, right_frame]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            processed_frames = list(executor.map(self.process_frame, frames, self.camera_models))

        self.birdview.update_frames(processed_frames)
        self.birdview.make_luminance_balance().stitch_all_parts()
        self.birdview.make_white_balance()
        self.birdview.copy_car_image()
        return self.birdview.image