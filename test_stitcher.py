"""
此脚本用于测试 BirdViewStitcher 模块。
"""
import cv2
import time
import os
import glob
from surround_view.stitcher_module import BirdViewStitcher

# 输入和输出目录
base_input_dir = r"E:\zhiwei\1\zhiwei-adas360\surround-view-system-introduction\camera_captures"
output_dir = r"E:\zhiwei\1\zhiwei-adas360\surround-view-system-introduction\camera_captures\stitched_images"
os.makedirs(output_dir, exist_ok=True)

# 相机名称和对应的子目录
camera_info = {
    "front": os.path.join(base_input_dir, "camera_1"),
    "back": os.path.join(base_input_dir, "camera_2"),
    "left": os.path.join(base_input_dir, "camera_3"),
    "right": os.path.join(base_input_dir, "camera_4")
}
# 提前加载一组初始化图像用于 BirdViewStitcher
init_images = []
for cam in ["front", "back", "left", "right"]:
    # 从该相机目录下任意选一张图作为初始化图像
    files = sorted(glob.glob(os.path.join(camera_info[cam], "*.jpg")))
    if not files:
        raise RuntimeError(f"❌ 初始化失败，找不到 {cam} 的图像")
    init_img = cv2.imread(files[0])
    init_images.append(init_img)

# 创建拼接器，预加载这组图像
stitcher = BirdViewStitcher(init_images=init_images)
# 创建拼接器实例
#stitcher = BirdViewStitcher()

# 最大等待次数和每次等待时间
MAX_WAIT_TIMES = 5
WAIT_INTERVAL = 2

def get_image_number(file_name):
    """从文件名中提取序号"""
    parts = file_name.split('_')
    if len(parts) > 1:
        try:
            return int(parts[-1].split('.')[0])
        except ValueError:
            pass
    return None

def get_image_groups():
    """获取按序号分组的图片"""
    number_dict = {}
    for camera_name, camera_dir in camera_info.items():
        if os.path.exists(camera_dir):
            image_files = glob.glob(os.path.join(camera_dir, '*.png')) + glob.glob(os.path.join(camera_dir, '*.jpg')) + glob.glob(os.path.join(camera_dir, '*.jpeg'))
            for file in sorted(image_files):  # 按文件名排序
                file_name = os.path.basename(file)
                if file_name.lower().startswith(camera_name.lower()):
                    number = get_image_number(file_name)
                    if number:
                        if number not in number_dict:
                            number_dict[number] = {}
                        number_dict[number][camera_name] = file
    return number_dict

def is_file_ready(file_path):
    """检查文件是否完整写入"""
    try:
        file_size_1 = os.path.getsize(file_path)
        time.sleep(0.1)
        file_size_2 = os.path.getsize(file_path)
        return file_size_1 == file_size_2
    except Exception:
        return False

image_groups = get_image_groups()
processed = False
for number in sorted(image_groups.keys()):
    wait_times = 0
    while wait_times < MAX_WAIT_TIMES:
        group = image_groups.get(number, {})
        if len(group) == 4:  # 检查是否有四个相机的图片
            frames = []
            valid_group = True
            for camera_name in ["front", "back", "left", "right"]:
                file_path = group[camera_name]
                if is_file_ready(file_path):
                    frame = cv2.imread(file_path)
                    if frame is not None:
                        frames.append(frame)
                        # 删除已处理的图片，避免重复处理
                        os.remove(file_path)
                    else:
                        valid_group = False
                        break
                else:
                    valid_group = False
                    break

            if valid_group and len(frames) == 4:
                start_time = time.time()  # 记录循环开始时间

                # 拼接图像
                stitched_image = stitcher.stitch_frames(*frames)

                # 生成输出文件名
                output_filename = f"stitched_{number}.png"
                output_path = os.path.join(output_dir, output_filename)

                # 保存拼接后的图像
                cv2.imwrite(output_path, stitched_image)

                end_time = time.time()  # 记录循环结束时间
                loop_time = end_time - start_time  # 计算循环耗时
                print(f"本次循环耗时: {loop_time:.4f} 秒")
                processed = True
            break
        else:
            wait_times += 1
            time.sleep(WAIT_INTERVAL)
            image_groups = get_image_groups()

    if wait_times == MAX_WAIT_TIMES:
        pass

    if processed:
        break

print("处理完成，程序退出。")