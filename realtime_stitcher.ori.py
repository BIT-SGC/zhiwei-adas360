import cv2
import numpy as np
import threading
import time
import os
import shutil
import subprocess
from datetime import datetime
from queue import Queue
from surround_view.stitcher_module import BirdViewStitcher
# 导入 glob 模块
import glob

# =============== 全局变量声明 ===============
running = True
capture_flag = False  
show_stitched = True   
last_capture_status = ""
test_stitcher_process = None
# 定义全局锁
save_lock = threading.Lock()
# ==========================================

# =============== 配置区域 ===============
CAMERA_URLS = [
    'rtsp://192.168.1.40:554/stream_0',
    'rtsp://192.168.1.41:554/stream_0',
    'rtsp://192.168.1.42:554/stream_0',
    'rtsp://192.168.1.43:554/stream_0'
]

SAVE_BASE_DIR = r"E:\zhiwei\1\zhiwei-adas360\surround-view-system-introduction\camera_captures"
MAX_AUTO_CAPTURES = 200              
MAX_CAPTURE_DURATION = 3600          
# 设置自动抓拍间隔为 10 秒
AUTO_CAPTURE_INTERVAL = 2         
MIN_DISK_SPACE = 1024                # 最小保留磁盘空间(MB)

# 相机名称映射
CAMERA_NAMES = ["front", "back", "left", "right"]
# ======================================

class ImageSaver(threading.Thread):
    """独立的图像保存线程"""
    def __init__(self):
        super().__init__()
        self.queue = Queue(maxsize=100)
        self.running = True
        self.daemon = True
        self.total_saved = 0
        
    def run(self):
        global capture_flag
        while self.running or not self.queue.empty():
            try:
                task = self.queue.get(timeout=1)
                camera_id, frame, filename = task
                
                if not self.check_disk_space():
                    print("⚠️ 磁盘空间不足，停止保存")
                    capture_flag = False
                    self.queue.task_done()
                    continue
                
                try:
                    success = cv2.imwrite(filename, frame)
                    if success:
                        self.total_saved += 1
                        print(f"📸📸 相机 {camera_id} 保存成功 ({self.total_saved}张)")
                    else:
                        print(f"❌❌ 相机 {camera_id} 保存失败")
                except Exception as e:
                    print(f"⚠️ 相机 {camera_id} 保存异常: {str(e)}")
                    
                self.queue.task_done()
            except:
                pass

    def add_task(self, camera_id, frame, filename):
        """添加保存任务到队列"""
        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except:
                pass
        self.queue.put((camera_id, frame, filename))
        
    def check_disk_space(self):
        """检查磁盘剩余空间"""
        if MIN_DISK_SPACE <= 0:
            return True
            
        total, used, free = shutil.disk_usage(SAVE_BASE_DIR)
        free_mb = free // (1024 * 1024)
        return free_mb >= MIN_DISK_SPACE
        
    def stop(self):
        """停止线程"""
        self.running = False

class CameraThread(threading.Thread):
    def __init__(self, url, camera_id, image_saver, camera_threads):
        super().__init__()
        self.url = url
        self.camera_id = camera_id
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
        self.frame_count = 0
        self.image_saver = image_saver
        self.last_capture_time = 0
        self.auto_capture_count = 0
        self.auto_capture_start_time = 0
        self.save_count = 0  # 新增保存计数
        self.camera_threads = camera_threads  # 保存 camera_threads 列表

        # 根据相机 ID 确定保存目录
        camera_name = CAMERA_NAMES[camera_id]
        self.save_dir = os.path.join(SAVE_BASE_DIR, f"camera_{camera_id + 1}")
        os.makedirs(self.save_dir, exist_ok=True)

    def run(self):
        global capture_flag, last_capture_status
        self._initialize_capture()

        while self.running and running:
            start_time = time.time()

            frame = self._read_frame()

            with self.lock:
                if frame is not None:
                    self.current_frame = frame
                    self.last_valid_frame = frame
                    self.frame_count += 1

                    # 自动捕获逻辑
                    if capture_flag:
                        current_time = time.time()

                        # 初始化抓拍计时
                        if self.auto_capture_start_time == 0:
                            self.auto_capture_start_time = current_time
                            self.auto_capture_count = 0
                            print(f"📡📡 相机 {self.camera_id + 1} 开始自动抓拍")

                        # 检查抓拍条件
                        should_capture = (
                            (MAX_AUTO_CAPTURES <= 0 or self.auto_capture_count < MAX_AUTO_CAPTURES) and
                            (MAX_CAPTURE_DURATION <= 0 or 
                             (current_time - self.auto_capture_start_time) < MAX_CAPTURE_DURATION) and
                            (current_time - self.last_capture_time) >= AUTO_CAPTURE_INTERVAL
                        )

                        if should_capture:
                            self.last_capture_time = current_time
                            self.auto_capture_count += 1
                            self._enqueue_save_task()

                            # 更新状态信息
                            duration = current_time - self.auto_capture_start_time
                            remaining = MAX_CAPTURE_DURATION - duration if MAX_CAPTURE_DURATION > 0 else float('inf')

                            last_capture_status = (
                                f"抓拍中: {self.auto_capture_count}/{MAX_AUTO_CAPTURES if MAX_AUTO_CAPTURES > 0 else '∞'}张 | "
                                f"剩余时间: {max(0, int(remaining))}秒"
                            )

                            # 检查是否所有相机都完成一次抓拍
                            all_cameras_saved = all(
                                thread.save_count == self.save_count for thread in self.camera_threads  # 使用 self.camera_threads
                            )
                            if all_cameras_saved:
                                # 启动 test_stitcher.py
                                try:
                                    subprocess.Popen(['python', 'test_stitcher.py'])
                                    print("已启动 test_stitcher.py 处理当前抓拍组")
                                except Exception as e:
                                    print(f"启动 test_stitcher.py 失败: {e}")

                        # 检查停止条件
                        stop_condition = (
                            (MAX_AUTO_CAPTURES > 0 and self.auto_capture_count >= MAX_AUTO_CAPTURES) or
                            (MAX_CAPTURE_DURATION > 0 and 
                             (current_time - self.auto_capture_start_time) >= MAX_CAPTURE_DURATION)
                        )

                        if stop_condition and capture_flag:
                            capture_flag = False
                            duration = current_time - self.auto_capture_start_time
                            print(f"🛑🛑🛑 相机 {self.camera_id + 1} 自动抓拍完成，"
                                  f"共抓拍 {self.auto_capture_count} 张，耗时 {duration:.1f} 秒")

            # 计算 FPS
            self.fps_counter += 1
            elapsed = time.time() - self.last_fps_time
            if elapsed >= 1.0:
                self.fps = self.fps_counter / elapsed
                self.fps_counter = 0
                self.last_fps_time = time.time()

            # 保持稳定帧率
            process_time = time.time() - start_time
            sleep_time = max(0.01, 0.03 - process_time)
            time.sleep(sleep_time)

        if self.cap is not None:
            self.cap.release()

    def _enqueue_save_task(self):
        """将保存任务加入队列"""
        if self.current_frame is None and self.last_valid_frame is None:
            return
        self.save_count += 1
        camera_name = CAMERA_NAMES[self.camera_id]
        filename = os.path.join(self.save_dir, f"{camera_name}_{self.save_count}.png")
        frame = self.current_frame if self.current_frame is not None else self.last_valid_frame

        self.image_saver.add_task(self.camera_id + 1, frame, filename)

    def _initialize_capture(self):
        """初始化视频捕获"""
        print(f"🔌🔌 正在连接相机 {self.camera_id + 1}: {self.url}")
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            print(f"  ❌❌ 连接失败，稍后重试...")
            self.cap = None
            return False
        print(f"  ✅ 相机 {self.camera_id + 1} 连接成功")
        return True
    
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
            print(f"⚠️ 相机 {self.camera_id + 1} 读取错误：{str(e)}")
        
        self.consecutive_fails += 1
        print(f"⚠️ 相机 {self.camera_id + 1} 失败 ({self.consecutive_fails}/{self.max_fails})")
        
        if self.consecutive_fails >= self.max_fails:
            print(f"🔁🔁 相机 {self.camera_id + 1} 重新连接...")
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
    global test_stitcher_process, global_save_timestamp
    print("📸📸 正在手动保存所有相机画面...")
    with save_lock:
        global_save_timestamp = None
    for thread in camera_threads:
        thread._enqueue_save_task()
    print("✅ 保存任务已提交")
    # 启动 test_stitcher.py
    test_stitcher_process = subprocess.Popen(['python', 'test_stitcher.py'])
    with save_lock:
        global_save_timestamp = None

def create_stitched_image(frames, camera_threads):
    """创建拼接后的图像"""
    if not frames or len(frames) != 4:
        return None
    
    # 调整所有帧到相同尺寸
    target_size = None
    for frame in frames:
        if frame is not None and frame.size > 0:
            target_size = (frame.shape[1] // 2, frame.shape[0] // 2)
            break
    
    if target_size is None:
        target_size = (640, 480)
    
    # 调整所有帧大小
    resized_frames = []
    for i, frame in enumerate(frames):
        if frame is not None and frame.size > 0:
            resized = cv2.resize(frame, target_size)
        else:
            resized = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            cv2.putText(resized, f"Cam {i+1} Offline", 
                       (10, target_size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        resized_frames.append(resized)
    
    # 创建拼接图像
    top_row = np.hstack((resized_frames[0], resized_frames[1]))
    bottom_row = np.hstack((resized_frames[2], resized_frames[3]))
    stitched = np.vstack((top_row, bottom_row))
    
    # 添加分隔线
    color = (0, 255, 0)
    thickness = 2
    cv2.line(stitched, (target_size[0], 0), (target_size[0], stitched.shape[0]), color, thickness)
    cv2.line(stitched, (0, target_size[1]), (stitched.shape[1], target_size[1]), color, thickness)
    
    # 添加相机信息
    for i in range(4):
        x = (i % 2) * target_size[0] + 10
        y = (i // 2) * target_size[1] + 30
        cv2.putText(stitched, f"Cam {i+1} ({camera_threads[i].fps:.1f}FPS)", 
                   (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # 添加全局状态信息
    status_lines = [
        f"模式: {'自动抓拍' if capture_flag else '手动抓拍'}",
        f"操作: S=保存 A=自动 T=切换 F=全屏 ESC=退出"
    ]
    
    if capture_flag:
        status_lines.append(last_capture_status)
    
    for i, line in enumerate(status_lines):
        y_pos = stitched.shape[0] - 30 - (len(status_lines)-1-i)*30
        cv2.putText(stitched, line, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return stitched

def main():
    global running, capture_flag, show_stitched, test_stitcher_process
    
    # 创建图像保存线程
    image_saver = ImageSaver()
    image_saver.start()
    
    # 创建相机保存目录
    os.makedirs(SAVE_BASE_DIR, exist_ok=True)

    # 创建相机线程
    camera_threads = []
    for idx, url in enumerate(CAMERA_URLS):
        thread = CameraThread(url, idx, image_saver, camera_threads)  # 传入 camera_threads 列表
        thread.daemon = True
        camera_threads.append(thread)
        thread.start()

    # 定义 camera_info 变量
    camera_info = {
        "front": os.path.join(SAVE_BASE_DIR, "camera_1"),
        "back": os.path.join(SAVE_BASE_DIR, "camera_2"),
        "left": os.path.join(SAVE_BASE_DIR, "camera_3"),
        "right": os.path.join(SAVE_BASE_DIR, "camera_4")
    }

    # 等待相机线程保存初始图像
    max_wait_time = 10  # 最大等待时间（秒）
    wait_start_time = time.time()
    all_images_found = False

    while time.time() - wait_start_time < max_wait_time and not all_images_found:
        all_images_found = True
        for cam in ["front", "back", "left", "right"]:
            files = sorted(glob.glob(os.path.join(camera_info[cam], "*.png")))
            if not files:
                all_images_found = False
                break
        if not all_images_found:
            time.sleep(1)

    if not all_images_found:
        print("⚠️ 等待超时，部分相机未保存初始图像，尝试手动保存一次")
        save_all_cameras(camera_threads)
        time.sleep(2)  # 等待保存完成

    # 初始化 BirdViewStitcher
    init_images = []
    for cam in ["front", "back", "left", "right"]:
        files = sorted(glob.glob(os.path.join(camera_info[cam], "*.png")))
        if not files:
            print(f"❌ 初始化失败，找不到 {cam} 的图像，跳过此相机初始化")
            init_images.append(np.zeros((480, 640, 3), dtype=np.uint8))  # 使用空白图像替代
            continue
        init_img = cv2.imread(files[0])
        if init_img is None:
            print(f"❌ 初始化失败，{cam} 的第一张图读取失败：{files[0]}，跳过此相机初始化")
            init_images.append(np.zeros((480, 640, 3), dtype=np.uint8))  # 使用空白图像替代
            continue
        init_images.append(init_img)
    stitcher = BirdViewStitcher(init_images=init_images)

    try:
        # 等待所有相机初始化
        time.sleep(1)
        
        print("\n" + "="*50)
        print(f"📁📁 图像保存目录: {os.path.abspath(SAVE_BASE_DIR)}")
        print(f"🔢🔢 自动抓拍设置: 最多{MAX_AUTO_CAPTURES}张/相机, 最长{MAX_CAPTURE_DURATION}秒")
        print("🎮🎮 操作指南:")
        print("  S - 手动保存当前画面")
        print("  A - 开启/关闭自动抓拍")
        print("  T - 切换显示模式(分屏/拼接)")
        print("  F - 全屏切换")
        print("ESC - 退出程序")
        print("="*50 + "\n")
        
        # 主显示循环
        while running:
            start_time = time.time()
            frames = []
            
            # 从每个相机获取最新帧
            for thread in camera_threads:
                frames.append(thread.get_current_frame())
            
            # 生成鸟瞰图
            if len(frames) == 4:
                birdview_image = stitcher.stitch_frames(*frames)
                if birdview_image is not None:
                    cv2.imshow("Bird's Eye View", cv2.resize(birdview_image, (600, 800)))
            
            # 处理键盘输入
            key = cv2.waitKey(1)
            if key == 27:  # ESC 退出
                running = False
            elif key == ord('s') or key == ord('S'):  # 手动保存
                save_all_cameras(camera_threads)
            elif key == ord('a') or key == ord('A'):  # 切换自动保存模式
                capture_flag = not capture_flag
                print(f"🔄🔄 自动抓拍模式 {'开启' if capture_flag else '关闭'}")
            elif key == ord('t') or key == ord('T'):  # 切换显示模式
                show_stitched = not show_stitched
                if show_stitched:
                    cv2.destroyWindow("四路监控系统 - 分屏模式")
                    print("🖥🖥🖥️ 切换到拼接显示模式")
                else:
                    cv2.destroyWindow("四路监控系统 - 拼接模式")
                    print("🖥🖥🖥️ 切换到分屏显示模式")
            elif key == ord('f') or key == ord('F'):  # 全屏切换
                window_name = "Bird's Eye View"
                fullscreen = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                     not fullscreen)
            
            # 控制帧率
            process_time = time.time() - start_time
            sleep_time = max(0.001, 0.03 - process_time)
            time.sleep(sleep_time)

    finally:
        # 清理资源
        running = False
        capture_flag = False
        time.sleep(0.5)
        
        for thread in camera_threads:
            thread.stop()
        
        image_saver.stop()
        image_saver.join()

        if test_stitcher_process:
            test_stitcher_process.terminate()
        
        cv2.destroyAllWindows()
        print("🛑🛑🛑 系统已安全关闭")
        print(f"💾💾 本次运行共保存 {image_saver.total_saved} 张图片")

if __name__ == "__main__":
    main()