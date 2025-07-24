import cv2
import numpy as np
import threading
import time
import os
from datetime import datetime

# 相机配置
CAMERA_URLS = [
    'rtsp://192.168.1.40:554/stream_0',
    'rtsp://192.168.1.41:554/stream_0',
    'rtsp://192.168.1.42:554/stream_0',
    'rtsp://192.168.1.43:554/stream_0'
]

# 保存路径配置
SAVE_BASE_DIR = "./camera_captures"  # 图片保存根目录

# 全局控制变量
running = True
capture_flag = False  # 图片捕获标志

class CameraThread(threading.Thread):
    def __init__(self, url, camera_id):
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
        self.frame_count = 0  # 帧计数器
        self.save_dir = os.path.join(SAVE_BASE_DIR, f"camera_{camera_id+1}")  # 每个相机的保存目录
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
    def run(self):
        self._initialize_capture()
        
        while self.running and running:
            start_time = time.time()
            
            # 尝试读取帧
            frame = self._read_frame()
            
            # 更新当前帧
            with self.lock:
                if frame is not None:
                    self.current_frame = frame
                    self.last_valid_frame = frame
                    self.frame_count += 1
                    
                    # 自动捕获模式下保存帧
                    if capture_flag:
                        self.save_current_frame()
            
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
        
        # 清理资源
        if self.cap is not None:
            self.cap.release()
    
    def _initialize_capture(self):
        """初始化视频捕获"""
        print(f"初始化相机 {self.camera_id + 1}: {self.url}")
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            print(f"  ⚠️ 连接失败，稍后重试...")
            self.cap = None
            return False
        return True
    
    def _read_frame(self):
        """读取一帧图像并处理错误"""
        # 连接丢失时尝试重连
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
        
        # 处理读取失败
        self.consecutive_fails += 1
        print(f"相机 {self.camera_id + 1} 失败 ({self.consecutive_fails}/{self.max_fails})")
        
        # 过多失败后重新连接
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
        
        # 没有有效帧时生成错误图像
        error_img = self._create_error_image()
        return error_img
    
    def save_current_frame(self):
        """保存当前帧到文件（线程安全）"""
        with self.lock:
            if self.current_frame is None and self.last_valid_frame is None:
                return False
            
            # 使用时间戳和帧计数生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_dir, f"cam_{self.camera_id+1}_{timestamp}_{self.frame_count}.jpg")
            
            # 保存图像
            frame = self.current_frame if self.current_frame is not None else self.last_valid_frame
            try:
                success = cv2.imwrite(filename, frame)
                if success:
                    print(f"相机 {self.camera_id+1} 保存成功：{filename}")
                    return True
                print(f"相机 {self.camera_id+1} 保存失败：{filename}")
                return False
            except Exception as e:
                print(f"相机 {self.camera_id+1} 保存异常：{str(e)}")
                return False
    
    def _create_error_image(self):
        """创建错误提示图像"""
        error_img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.putText(error_img, f"Error Cam {self.camera_id + 1}", 
                   (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return error_img
    
    def stop(self):
        """停止线程"""
        self.running = False

def save_all_cameras(camera_threads):
    """保存所有相机的当前帧"""
    print("正在保存所有相机画面...")
    for thread in camera_threads:
        thread.save_current_frame()
    print("保存完成！")

# 创建相机保存目录
os.makedirs(SAVE_BASE_DIR, exist_ok=True)

# 创建相机线程
camera_threads = []
for idx, url in enumerate(CAMERA_URLS):
    thread = CameraThread(url, idx)
    thread.daemon = True  # 主线程退出时自动结束
    camera_threads.append(thread)
    thread.start()

try:
    # 等待所有相机初始化
    time.sleep(1)
    
    print(f"图像将保存到：{os.path.abspath(SAVE_BASE_DIR)}")
    print("操作指南：")
    print("  s - 保存当前所有相机画面")
    print("  a - 开启/关闭自动连续抓拍模式")
    print("  f - 切换全屏显示")
    print("ESC - 退出程序")
    
    # 主显示循环
    while running:
        start_time = time.time()
        frames = []
        
        # 从每个相机获取最新帧
        for thread in camera_threads:
            frames.append(thread.get_current_frame())
        
        # 创建显示画面 (2x2 网格)
        if frames and frames[0] is not None:
            grid_height = frames[0].shape[0] // 8
            grid_width = frames[0].shape[1] // 8
        else:
            grid_height, grid_width = 480, 640  # 默认尺寸
        
        grid_size = (grid_height * 2, grid_width * 2, 3)
        display_frame = np.zeros(grid_size, dtype=np.uint8)
        
        # 将帧排列到网格中
        positions = [
            (0, 0), (0, grid_width),
            (grid_height, 0), (grid_height, grid_width)
        ]
        
        for i, frame in enumerate(frames):
            if frame is not None:
                resized = cv2.resize(frame, (grid_width, grid_height))
            else:
                resized = camera_threads[i]._create_error_image()
                resized = cv2.resize(resized, (grid_width, grid_height))
            
            y, x = positions[i]
            display_frame[y:y+grid_height, x:x+grid_width] = resized
            
            # 在每路画面上添加相机标识
            cv2.putText(display_frame, f"Cam {i + 1} ({camera_threads[i].fps:.1f}FPS)", 
                       (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # # 添加全局状态信息
        # status_text = f"模式：{'连续抓拍' if capture_flag else '单次抓拍'} | 按 S 保存图像"
        # cv2.putText(display_frame, status_text, (10, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 显示总画面
        cv2.imshow("Quad Camera Display", display_frame)
        
        # 处理键盘输入
        key = cv2.waitKey(1)
        if key == 27:  # ESC 退出
            running = False
        elif key == ord('s'):  # 手动保存
            save_all_cameras(camera_threads)
        elif key == ord('a'):  # 切换自动保存模式
            capture_flag = not capture_flag
            print(f"自动抓拍模式 {'开启' if capture_flag else '关闭'}")
        elif key == ord('f'):  # 全屏切换
            fullscreen = cv2.getWindowProperty("Quad Camera Display", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Quad Camera Display", cv2.WND_PROP_FULLSCREEN, 
                                 not fullscreen)
        
        # 控制帧率
        process_time = time.time() - start_time
        sleep_time = max(0.001, 0.03 - process_time)
        time.sleep(sleep_time)

finally:
    # 清理资源
    running = False
    capture_flag = False
    time.sleep(0.5)  # 给线程退出时间
    
    for thread in camera_threads:
        thread.stop()
    
    cv2.destroyAllWindows()
    print("系统已安全关闭")