import os
import argparse
import numpy as np
import cv2
from PIL import Image
from surround_view import FisheyeCameraModel, display_image, BirdView
import surround_view.param_settings as settings
# 导入 datetime 模块
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="抓拍图片输入目录")
    parser.add_argument("--output_dir", required=True, help="拼接后图片输出目录")
    args = parser.parse_args()

    names = settings.camera_names
    yamls = [os.path.join(os.getcwd(), "yaml", name + ".yaml") for name in names]
    camera_models = [FisheyeCameraModel(camera_file, camera_name) for camera_file, camera_name in zip (yamls, names)]

    projected = []
    for name, camera in zip(names, camera_models):
        # 查找对应相机的最新抓拍图片
        image_files = [f for f in os.listdir(args.input_dir) if f.startswith(f"cam_{names.index(name)+1}_")]
        if not image_files:
            print(f"未找到 {name} 相机的抓拍图片")
            return
        latest_image_file = max(image_files, key=lambda x: os.path.getmtime(os.path.join(args.input_dir, x)))
        image_path = os.path.join(args.input_dir, latest_image_file)
        img = cv2.imread(image_path)
        img = camera.undistort(img)
        img = camera.project(img)
        img = camera.flip(img)
        projected.append(img)

    birdview = BirdView()
    Gmat, Mmat = birdview.get_weights_and_masks(projected)
    birdview.update_frames(projected)
    birdview.make_luminance_balance().stitch_all_parts()
    birdview.make_white_balance()
    birdview.copy_car_image()
    ret = display_image("BirdView Result", birdview.image)
    if ret > 0:
        Image.fromarray((Gmat * 255).astype(np.uint8)).save(os.path.join(args.output_dir, "weights.png"))
        Image.fromarray(Mmat.astype(np.uint8)).save(os.path.join(args.output_dir, "masks.png"))
    # 保存拼接后的鸟瞰图
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stitched_image_path = os.path.join(args.output_dir, f"stitched_{timestamp}.png")
    cv2.imwrite(stitched_image_path, birdview.image)
    print(f"拼接后的鸟瞰图已保存到: {stitched_image_path}")

if __name__ == "__main__":
    main()
