import cv2
import time
import os
import glob
from surround_view.stitcher_module import BirdViewStitcher

base_input_dir = r"E:\zhiwei\1\zhiwei-adas360\surround-view-system-introduction\camera_captures"
output_dir = os.path.join(base_input_dir, "stitched_images")
os.makedirs(output_dir, exist_ok=True)

camera_info = {
    "front": os.path.join(base_input_dir, "camera_1"),
    "back":  os.path.join(base_input_dir, "camera_2"),
    "left":  os.path.join(base_input_dir, "camera_3"),
    "right": os.path.join(base_input_dir, "camera_4"),
}

# ----------- 初始化 BirdViewStitcher -----------
init_images = []
for cam in ["front", "back", "left", "right"]:
    files = sorted(glob.glob(os.path.join(camera_info[cam], "*.png")))
    print(os.path.join(camera_info[cam], "*.png"))
    if not files:
        raise RuntimeError(f"❌ 初始化失败，找不到 {cam} 的图像")
    init_img = cv2.imread(files[0])
    if init_img is None:
        raise RuntimeError(f"❌ 初始化失败，{cam} 的第一张图读取失败：{files[0]}")
    init_images.append(init_img)

stitcher = BirdViewStitcher(init_images=init_images)

MAX_WAIT_TIMES = 5
WAIT_INTERVAL = 2.0

def get_image_number(file_name: str):
    parts = file_name.split('_')
    if len(parts) > 1:
        try:
            return int(parts[-1].split('.')[0])
        except ValueError:
            return None
    return None

def get_image_groups():
    number_dict = {}
    for camera_name, camera_dir in camera_info.items():
        if not os.path.exists(camera_dir):
            continue
        exts = ('*.png', '*.jpg', '*.jpeg')
        image_files = []
        for e in exts:
            image_files.extend(glob.glob(os.path.join(camera_dir, e)))
        for file in sorted(image_files):
            file_name = os.path.basename(file)
            if file_name.lower().startswith(camera_name.lower()):
                number = get_image_number(file_name)
                if number is not None:
                    number_dict.setdefault(number, {})[camera_name] = file
    #import pdb;pdb.set_trace()
    return number_dict

def is_file_ready(file_path):
    try:
        size1 = os.path.getsize(file_path)
        time.sleep(0.2)
        size2 = os.path.getsize(file_path)
        return size1 == size2
    except Exception:
        return False

def read_four_frames(group_paths):
    """按固定顺序读取四张图像，全部成功才返回列表，否则返回 None"""
    frames = []
    for cam in ["front", "back", "left", "right"]:
        path = group_paths.get(cam)
        if path is None or (not is_file_ready(path)):
            return None
        img = cv2.imread(path)
        if img is None:
            return None
        frames.append(img)
    return frames

# ----------- 主循环 -----------
while True:
    image_groups = get_image_groups()
    print(image_groups)
    if not image_groups:
        time.sleep(WAIT_INTERVAL)
        continue

    processed_any = False
    for number in sorted(image_groups.keys()):
        group_paths = image_groups[number]
        if len(group_paths) < 4:
            # 该组还没齐
            continue

        wait_times = 0
        frames = read_four_frames(group_paths)
        while frames is None and wait_times < MAX_WAIT_TIMES:
            wait_times += 1
            time.sleep(WAIT_INTERVAL)
            frames = read_four_frames(group_paths)

        if frames is None:
            # 超时放弃该组，继续处理下一个
            continue

        # 真的有四张并读好了
        t0 = time.time()
        stitched = stitcher.stitch_frames(*frames)
        out_name = f"stitched_{number:06d}.png"
        out_path = os.path.join(output_dir, out_name)
        ok = cv2.imwrite(out_path, stitched)
        dt = time.time() - t0
        print(f"[{number}] 拼接完成，耗时 {dt:.4f}s，保存结果：{ok}")

        # 成功后再删除原图
        # for cam in ["cam_1", "cam_2", "cam_3", "cam_4"]:
        #     try:
        #         os.remove(group_paths[cam])
        #     except OSError:
        #         pass

        processed_any = True

    # 如果这一轮什么都没处理，说明当前没有新图了，休眠一会继续
    if not processed_any:
        time.sleep(WAIT_INTERVAL)