import cv2
import numpy as np
import threading
import time
import os
import shutil
from datetime import datetime
from queue import Queue
import subprocess

# =============== 全局变量声明 ===============
running = True
capture_flag = False  
show_stitched = True   
last_capture_status = ""
# ==========================================

# =============== 配置区域 ===============
CAMERA_URLS = [
    'rtsp://192.168.1.40:554/stream_0',
    'rtsp://192.168.1.41:554/stream_0',
    'rtsp://192.168.1.42:554/stream_0',
    'rtsp://192.168.1.43:554/stream_0'
]

SAVE_BASE_DIR = "./camera_captures"
MAX_AUTO_CAPTURES = 200              
MAX_CAPTURE_DURATION = 3600          
AUTO_CAPTURE_INTERVAL = 0.5          
MIN_DISK_SPACE = 1024                # 最小保留磁盘空间(MB)
# 抓拍图片保存目录
CAPTURE_IMAGES_DIR = os.path.join(SAVE_BASE_DIR, "captured_images")
# 拼接后图片保存目录
STITCHED_IMAGES_DIR = os.path.join(SAVE_BASE_DIR, "stitched_images")
# ======================================

class ImageSaver(threading.Thread):
    """独立的图像保存线程"""
    def __init__(self):
        super().__init__()
        self.running = True
        self.task_queue = Queue()
        os.makedirs(CAPTURE_IMAGES_DIR, exist_ok=True)
        os.makedirs(STITCHED_IMAGES_DIR, exist_ok=True)

    def run(self):
        while self.running:
            if not self.task_queue.empty():
                camera_id, frame, filename = self.task_queue.get()
                if self.check_disk_space():
                    save_path = os.path.join(CAPTURE_IMAGES_DIR, filename)
                    cv2.imwrite(save_path, frame)
                    print(f"相机 {camera_id} 保存成功: {save_path}")
                    # 检查是否为连续抓拍模式，触发拼接
                    if capture_flag:
                        self.stitch_captured_images()
                else:
                    print("磁盘空间不足，停止保存")

    def add_task(self, camera_id, frame, filename):
        """添加保存任务到队列"""
        self.task_queue.put((camera_id, frame, filename))

    def check_disk_space(self):
        """检查磁盘剩余空间"""
        total, used, free = shutil.disk_usage("/")
        free_mb = free // (2**20)
        return free_mb > MIN_DISK_SPACE

    def stop(self):
        """停止线程"""
        self.running = False

    def stitch_captured_images(self):
        """调用 run_get_weight_matrices.py 进行拼接"""
        try:
            subprocess.run(["python", "run_get_weight_matrices.py", "--input_dir", CAPTURE_IMAGES_DIR, "--output_dir", STITCHED_IMAGES_DIR], check=True)
            print("拼接完成并保存")
        except subprocess.CalledProcessError as e:
            print(f"拼接失败: {e}")

class CameraThread(threading.Thread):
    def __init__(self, url, camera_id, image_saver):
        super().__init__()
        self.url = url
        self.camera_id = camera_id
        self.image_saver = image_saver
        self.current_frame = None
        self.last_valid_frame = None
        self.cap = None
        self.lock = threading.Lock()
        self.running = True
        self.reconnect_interval = 2
        self.consecutive_fails = 0
        self.max_fails = 5
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.fps = 0
        self.frame_count = 0  # 帧计数器

    def _initialize_capture(self):
        """初始化视频捕获"""
        print(f"开始初始化相机 {self.camera_id + 1}: {self.url}")
        # 不指定后端，让 OpenCV 自动选择
        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            print(f"  ⚠️ 相机 {self.camera_id + 1} 连接失败，稍后重试...")
            self.cap = None
            return False
        print(f"  ✅ 相机 {self.camera_id + 1} 连接成功")
        return True

    def run(self):
        self._initialize_capture()
        while self.running and running:
            start_time = time.time()
            frame = self._read_frame()
            with self.lock:
                if frame is not None:
                    self.current_frame = frame
                    self.last_valid_frame = frame
                    self.frame_count += 1
                    if capture_flag:
                        self._enqueue_save_task()
            # 计算 FPS
            self.fps_counter += 1
            elapsed = time.time() - self.last_fps_time
            if elapsed >= 1.0:
                self.fps = self.fps_counter / elapsed
                self.fps_counter = 0
                self.last_fps_time = time.time()
            # 保持稳定帧率
            process_time = time.time() - start_time
            sleep_time = max(0.01, 0.03 - process_time)  # 目标 30FPS
            time.sleep(sleep_time)
        if self.cap is not None:
            self.cap.release()

    def _enqueue_save_task(self):
        """将保存任务加入队列"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cam_{self.camera_id+1}_{timestamp}.png"
        self.image_saver.add_task(self.camera_id, self.current_frame, filename)

    def _read_frame(self):
        """读取一帧图像并处理错误"""
        if self.cap is None:
            if time.time() - getattr(self, 'last_connect_attempt', 0) > self.reconnect_interval:
                self.last_connect_attempt = time.time()
                if self._initialize_capture():
                    return self._capture_single_frame()
            return None
        return self._capture_single_frame()

    def _capture_single_frame(self):
        """从打开的摄像头捕获一帧"""
        try:
            ret, frame = self.cap.read()
            if ret:
                self.consecutive_fails = 0
                return frame
        except Exception as e:
            print(f"相机 {self.camera_id + 1} 读取错误：{str(e)}")
        self.consecutive_fails += 1
        print(f"相机 {self.camera_id + 1} 失败 ({self.consecutive_fails}/{self.max_fails})")
        if self.consecutive_fails >= self.max_fails:
            print(f"相机 {self.camera_id + 1} 重新连接...")
            self.cap.release()
            self.cap = None
            self.consecutive_fails = 0
        return None

    def get_current_frame(self):
        """安全获取当前帧（线程安全）"""
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            elif self.last_valid_frame is not None:
                return self.last_valid_frame.copy()
        error_img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.putText(error_img, f"Error Cam {self.camera_id + 1}", 
                   (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return error_img

    def stop(self):
        """停止线程"""
        self.running = False

def save_all_cameras(camera_threads):
    """手动保存所有相机的当前帧"""
    print("正在保存所有相机画面...")
    for thread in camera_threads:
        thread._enqueue_save_task()
    print("保存完成！")

def create_stitched_image(frames, camera_threads):
    """创建拼接后的图像"""
    # 这里可以添加拼接逻辑
    pass

def main():
    global running, capture_flag
    image_saver = ImageSaver()
    image_saver.start()
    camera_threads = []
    for idx, url in enumerate(CAMERA_URLS):
        thread = CameraThread(url, idx, image_saver)
        thread.daemon = True
        camera_threads.append(thread)
        thread.start()
    time.sleep(1)
    print(f"图像将保存到：{os.path.abspath(SAVE_BASE_DIR)}")
    print("操作指南：")
    print("  s - 保存当前所有相机画面")
    print("  a - 开启/关闭自动连续抓拍模式")
    print("ESC - 退出程序")
    while running:
        frames = []
        for thread in camera_threads:
            frames.append(thread.get_current_frame())
        if show_stitched:
            stitched_image = create_stitched_image(frames, camera_threads)
            if stitched_image is not None:
                cv2.imshow("Stitched Image", stitched_image)
        key = cv2.waitKey(1)
        if key == 27:  # ESC 退出
            running = False
        elif key == ord('s'):  # 手动保存
            save_all_cameras(camera_threads)
        elif key == ord('a'):  # 切换自动保存模式
            capture_flag = not capture_flag
            print(f"自动抓拍模式 {'开启' if capture_flag else '关闭'}")
    image_saver.stop()
    image_saver.join()
    for thread in camera_threads:
        thread.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()