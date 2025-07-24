import cv2
import time
import numpy as np
from surround_view.stitcher_module import BirdViewStitcher  # 根据实际路径调整

# ==== 图像路径（你之前给的路径）====
base_path = "/Users/hanshengliang/Desktop/1/zhiwei-adas360/surround-view-system-introduction/images"
image_paths = {
    "front": f"{base_path}/front.png",
    "back": f"{base_path}/back.png",
    "left": f"{base_path}/left.png",
    "right": f"{base_path}/right.png",
}

# ==== 加载并可选下采样 ====
def load_images(resize_to=None):
    images = {}
    for name, path in image_paths.items():
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"❌ 无法加载图像: {path}")
        if resize_to:
            img = cv2.resize(img, resize_to)
        images[name] = img
        print(f"✅ 读取 {name} 图像，尺寸: {img.shape}")
    return images

# ==== 测试拼接耗时 ====
def test_stitch_time(images, num_trials=10):
    #stitcher = BirdViewStitcher()  # 假设映射在 __init__ 中预加载
    stitcher = BirdViewStitcher(init_images=[
        images["front"], images["back"], images["left"], images["right"]
    ])
    total_time = 0

    print("\n🚀 开始测试拼接时间...")
    for i in range(num_trials):
        t0 = time.perf_counter()
        stitched = stitcher.stitch_frames(
            images["front"], images["back"], images["left"], images["right"]
        )
        t1 = time.perf_counter()
        duration = t1 - t0
        total_time += duration
        print(f"[{i+1}] 拼接耗时: {duration:.4f} 秒")

    avg = total_time / num_trials
    print(f"\n🎯 平均拼接耗时: {avg:.4f} 秒（共测试 {num_trials} 次）")

    return stitched

# ==== 主程序 ====
if __name__ == "__main__":
    # 设置缩放比例（例如 (960, 540) 可大幅提速），None 表示不缩放
    resize_resolution =  None 

    print("📂 正在加载图像...")
    images = load_images(resize_to=resize_resolution)

    stitched = test_stitch_time(images, num_trials=10)

    # 可视化结果
    cv2.imshow("Stitched Result", stitched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()