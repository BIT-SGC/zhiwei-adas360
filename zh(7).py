import os
import numpy as np
import cv2
import threading
import time
from surround_view import FisheyeCameraModel, BirdView
import surround_view.param_settings as settings

# =============== 全局配置 ===============
# 相机顺序和URL配置
CAMERA_CONFIG = [
    {"name": "front", "url": "rtsp://192.168.1.40:554/stream0"},  # 1号相机
    {"name": "back",  "url": "rtsp://192.168.1.41:554/stream0"},  # 2号相机
    {"name": "left",  "url": "rtsp://192.168.1.42:554/stream0"},  # 3号相机
    {"name": "right", "url": "rtsp://192.168.1.43:554/stream0"}   # 4号相机
]

# 更新全局设置
settings.camera_names = [cam["name"] for cam in CAMERA_CONFIG]

# 调试模式
DEBUG_MODE = True
# ======================================

class RobustBirdView(BirdView):
    """增强鲁棒性的鸟瞰图生成器"""
    def get_weights_and_masks(self, frames):
        try:
            if DEBUG_MODE:
                print("[DEBUG] 计算权重和掩码...")
                for i, frame in enumerate(frames):
                    if frame is None:
                        print(f"  ❌ 帧 {i} 为None")
                    else:
                        print(f"  ✅ 帧 {i} 尺寸: {frame.shape}")
            
            # 原始处理逻辑
            G, M = super().get_weights_and_masks(frames)
            
            # 验证输出
            if G is None or M is None:
                raise ValueError("权重或掩码计算返回None")
                
            return G, M
        except Exception as e:
            print(f"⚠️ 权重掩码计算失败: {str(e)}")
            # 生成默认权重和掩码
            h, w = frames[0].shape[:2] if frames[0] is not None else (600, 800)
            default_weight = np.ones((h, w), dtype=np.float32)
            default_mask = np.ones((h, w), dtype=np.uint8)
            return default_weight, default_mask

class CameraThread(threading.Thread):
    """增强版的相机采集线程"""
    def __init__(self, camera_id, config):
        super().__init__()
        self.camera_id = camera_id
        self.name = config["name"]
        self.url = config["url"]
        self.frame = None
        self.last_valid_frame = None
        self.cap = None
        self.lock = threading.Lock()
        self.running = True
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # 相机模型路径
        self.yaml_path = os.path.join(
            os.path.dirname(__file__), 
            "yaml", 
            f"{self.name}.yaml"
        )
        
        # 初始化相机模型
        try:
            self.camera_model = FisheyeCameraModel(self.yaml_path, self.name)
            if DEBUG_MODE:
                print(f"✅ 相机 {self.camera_id+1}({self.name}) 模型加载成功")
        except Exception as e:
            print(f"❌ 相机 {self.camera_id+1} 模型加载失败: {str(e)}")
            self.camera_model = None

    def run(self):
        self._initialize_capture()
        
        while self.running:
            start_time = time.time()
            frame = self._read_frame()
            
            # 处理帧
            with self.lock:
                if frame is not None:
                    try:
                        if self.camera_model:
                            frame = self.camera_model.undistort(frame)
                            frame = self.camera_model.project(frame)
                            frame = self.camera_model.flip(frame)
                        self.frame = frame
                        self.last_valid_frame = frame
                    except Exception as e:
                        print(f"⚠️ 相机 {self.camera_id+1} 帧处理错误: {str(e)}")
                
                # 计算FPS
                self.frame_count += 1
                elapsed = time.time() - self.last_fps_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.last_fps_time = time.time()
            
            # 控制采集频率
            time.sleep(max(0, 0.03 - (time.time() - start_time)))

    def _initialize_capture(self):
        """初始化视频采集"""
        print(f"🔌 正在连接相机 {self.camera_id+1}({self.name})...")
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            print(f"  ❌ 连接失败")
            self.cap = None
        else:
            print(f"  ✅ 连接成功")

    def _read_frame(self):
        """读取一帧图像"""
        if self.cap is None:
            time.sleep(1)
            self._initialize_capture()
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                return frame
        except Exception as e:
            print(f"⚠️ 相机 {self.camera_id+1} 读取错误: {str(e)}")
        
        return None

    def get_current_frame(self):
        """获取当前帧（线程安全）"""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            elif self.last_valid_frame is not None:
                return self.last_valid_frame.copy()
        
        # 返回错误图像
        error_img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.putText(error_img, f"Cam {self.camera_id+1} Error", 
                   (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return error_img

    def stop(self):
        """停止线程"""
        self.running = False
        if self.cap is not None:
            self.cap.release()

def create_birdview(frames, camera_threads):
    """创建鸟瞰图（带完整错误处理）"""
    try:
        if DEBUG_MODE:
            print("\n[鸟瞰图生成] 输入帧状态:")
            for i, frame in enumerate(frames):
                status = "✅ 有效" if frame is not None else "❌ 无效"
                print(f"  相机 {i+1}({camera_threads[i].name}): {status}")

        # 预处理所有帧
        processed_frames = []
        for i, (frame, thread) in enumerate(zip(frames, camera_threads)):
            if frame is None:
                # 生成占位图
                frame = np.zeros((600, 800, 3), dtype=np.uint8)
                cv2.putText(frame, f"{thread.name} Offline", 
                           (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            processed_frames.append(frame)

        # 创建鸟瞰图
        birdview = RobustBirdView()
        birdview.get_weights_and_masks(processed_frames)
        birdview.update_frames(processed_frames)
        birdview.make_luminance_balance().stitch_all_parts()
        birdview.make_white_balance()
        
        if DEBUG_MODE:
            print("✅ 鸟瞰图生成成功")
        return birdview.image

    except Exception as e:
        print(f"❌ 鸟瞰图生成失败: {str(e)}")
        # 返回错误图像
        error_img = np.zeros((600, 600, 3), dtype=np.uint8)
        cv2.putText(error_img, "BirdView Error", (150, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return error_img

def main():
    # 定义运行状态变量
    running = True

    # 初始化所有相机线程
    camera_threads = []
    for i, config in enumerate(CAMERA_CONFIG):
        thread = CameraThread(i, config)
        thread.daemon = True
        camera_threads.append(thread)
        thread.start()

    # 等待初始化完成
    time.sleep(2)

    # 主显示循环
    while running:
        start_time = time.time()
        
        # 获取所有相机帧
        frames = [thread.get_current_frame() for thread in camera_threads]
        
        # 生成鸟瞰图
        birdview = create_birdview(frames, camera_threads)
        
        # 显示结果
        if birdview is not None:
            cv2.imshow("实时鸟瞰图监控系统", birdview)
        
        # 处理键盘输入
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC退出
            running = False
        elif key == ord('d'):  # 切换调试模式
            global DEBUG_MODE
            DEBUG_MODE = not DEBUG_MODE
            print(f"调试模式 {'开启' if DEBUG_MODE else '关闭'}")
        
        # 控制帧率
        time.sleep(max(0, 0.03 - (time.time() - start_time)))

    # 清理资源
    for thread in camera_threads:
        thread.stop()
        thread.join()
    
    cv2.destroyAllWindows()
    print("🛑 系统已安全关闭")

if __name__ == "__main__":
    print("==========================================")
    print("🚗 实时鸟瞰图监控系统 v2.0")
    
    # 安全打印相机配置
    config_list = []
    for i, cam in enumerate(CAMERA_CONFIG):
        config_list.append(f"{i+1}号={cam['name']}")
    print(f"📡 相机配置: {', '.join(config_list)}")
    
    print("🎮 操作指南:")
    print("  ESC - 退出程序")
    print("  D   - 切换调试模式")
    print("==========================================")
    
    main()