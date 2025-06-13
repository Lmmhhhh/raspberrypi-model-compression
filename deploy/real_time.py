import logging
import threading
import time
import json
from datetime import datetime
from collections import deque
import os

import psutil
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# 사용자 설정
MODEL_PATH = "/home/gram/project/cap/model/mobilenet_v2-b0353104.pth"
CLASSES_PATH = "/home/gram/project/cap/model/imagenet_classes.txt"
# IP 카메라 스트림 URL 예시
SOURCE = "http://192.168.0.103:8081/video"
# USB 웹캠 사용 시 SOURCE = 0

DURATION = 180        # 측정 지속 시간(초)
WARMUP_FRAMES = 20    # 워밍업 프레임 수
LOG_DIR = "/home/gram/project/cap/log"  # 로그 디렉토리

# 모델 이름 추출 (파일명에서)
MODEL_NAME = os.path.splitext(os.path.basename(MODEL_PATH))[0]

# 현재 시간을 포함한 로그 파일명 생성
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(LOG_DIR, f"{MODEL_NAME}_{current_time}.log")

# 로그 디렉토리가 없으면 생성
os.makedirs(LOG_DIR, exist_ok=True)

# 로깅 설정: 파일과 콘솔 동시에 출력
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, mode='a'),
        logging.StreamHandler()
    ]
)

# 로그 시작 메시지
logging.info(f"=== 새로운 추론 세션 시작 ===")
logging.info(f"모델: {MODEL_NAME}")
logging.info(f"로그 파일: {LOG_PATH}")

class RealtimeInferenceAnalyzer:
    def __init__(self, model_path, classes_path,
                 source, duration, warmup_frames=20, max_history=300):
        # 파라미터
        self.duration = duration
        self.source = source
        self.warmup_frames = warmup_frames
        self.model_name = MODEL_NAME  # 모델 이름 저장

        # 모델 로드
        torch.backends.quantized.engine = 'qnnpack'
        self.model = torch.jit.load(model_path)
        self.model.eval()

        # 클래스 로드
        self._load_classes(classes_path)

        # 전처리 설정
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 메트릭 저장용 버퍼
        self.frame_idx = 0
        self.fps_history = deque(maxlen=max_history)
        self.inference_times = deque(maxlen=max_history)
        self.confidence_history = deque(maxlen=max_history)
        self.cpu_history = deque(maxlen=max_history)
        self.ram_history = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)

        # 리소스 모니터링 제어
        self._monitoring = False

    def _load_classes(self, path):
        try:
            with open(path, 'r') as f:
                self.classes = [l.strip() for l in f]
        except FileNotFoundError:
            logging.warning("클래스 파일을 찾을 수 없습니다. 인덱스로 대체합니다.")
            self.classes = [f"Class_{i}" for i in range(1000)]

    def _monitor_resources(self):
        while self._monitoring:
            self.cpu_history.append(psutil.cpu_percent(interval=0.1))
            self.ram_history.append(psutil.virtual_memory().percent)
            time.sleep(0.1)

    def _init_camera(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logging.error(f"카메라(소스: {self.source})를 열 수 없습니다.")
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    def _process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            out = self.model(tensor)
        probs = F.softmax(out[0], dim=0)
        top_p, top_idx = torch.topk(probs, 1)
        return top_p.item() * 100, self.classes[top_idx[0].item()]

    def run(self, display=True):
        logging.info(f"워밍업 프레임 {self.warmup_frames}프레임 제외 후 측정 시작")
        logging.info(f"실시간 추론 시작: {self.duration}s, 소스: {self.source}")
        cap = self._init_camera()
        if cap is None:
            return

        # 리소스 모니터링 스레드 시작
        self._monitoring = True
        monitor_t = threading.Thread(target=self._monitor_resources)
        monitor_t.start()

        start = time.time()
        frame_count = 0
        last_fps_time = start
        last_log_time = start  # 주기적 로그 출력용

        try:
            while (time.time() - start) < self.duration:
                ret, frame = cap.read()
                if not ret:
                    logging.warning("프레임 읽기 실패")
                    break

                self.frame_idx += 1
                # 워밍업 프레임 스킵
                if self.frame_idx <= self.warmup_frames:
                    continue

                t0 = time.time()
                confidence, pred_cls = self._process_frame(frame)
                infer_ms = (time.time() - t0) * 1000

                # 메트릭 기록
                elapsed = time.time() - start
                self.inference_times.append(infer_ms)
                self.confidence_history.append(confidence)
                self.timestamps.append(elapsed)

                frame_count += 1
                if elapsed - (last_fps_time - start) >= 1.0:
                    fps = frame_count / (time.time() - last_fps_time)
                    self.fps_history.append(fps)
                    frame_count = 0
                    last_fps_time = time.time()

                # 10초마다 진행 상황 로그
                if time.time() - last_log_time >= 10:
                    if self.fps_history and self.cpu_history:
                        logging.info(
                            f"진행: {int(elapsed)}s/{self.duration}s | "
                            f"FPS: {self.fps_history[-1]:.1f} | "
                            f"추론: {infer_ms:.1f}ms | "
                            f"CPU: {self.cpu_history[-1]:.1f}%"
                        )
                    last_log_time = time.time()

                # 결과 화면 표시
                if display:
                    cv2.putText(frame, f"{pred_cls[:20]} {confidence:.1f}%", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"FPS:{self.fps_history[-1]:.1f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("Inference", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        logging.info("사용자 중단")
                        break

        finally:
            self._monitoring = False
            monitor_t.join()
            cap.release()
            cv2.destroyAllWindows()

        return self._save_results()

    def _save_results(self):
        if not self.inference_times:
            logging.info("수집된 데이터 없음")
            return None

        stats = {
            "model_name": self.model_name,
            "duration": round(self.timestamps[-1], 2),
            "measured_frames": len(self.inference_times),
            "fps": {
                "avg": round(np.mean(self.fps_history), 2),
                "min": round(np.min(self.fps_history), 2),
                "max": round(np.max(self.fps_history), 2),
                "std": round(np.std(self.fps_history), 2),
            },
            "inference_ms": {
                "avg": round(np.mean(self.inference_times), 2),
                "min": round(np.min(self.inference_times), 2),
                "max": round(np.max(self.inference_times), 2),
                "std": round(np.std(self.inference_times), 2),
            },
            "confidence": {
                "avg": round(np.mean(self.confidence_history), 2),
                "min": round(np.min(self.confidence_history), 2),
                "max": round(np.max(self.confidence_history), 2),
            },
            "cpu": {
                "avg": round(np.mean(self.cpu_history), 2),
                "max": round(np.max(self.cpu_history), 2),
            },
            "ram": {
                "avg": round(np.mean(self.ram_history), 2),
                "max": round(np.max(self.ram_history), 2),
            },
            "timestamp": datetime.now().isoformat()
        }

        # JSON 파일명도 모델명과 시간 포함
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{self.model_name}_results_{timestamp}.json"
        json_path = os.path.join(LOG_DIR, json_filename)
        
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 결과 요약을 로그에도 기록
        logging.info("=== 추론 결과 요약 ===")
        logging.info(f"모델: {self.model_name}")
        logging.info(f"측정 시간: {stats['duration']}초")
        logging.info(f"처리 프레임: {stats['measured_frames']}개")
        logging.info(f"평균 FPS: {stats['fps']['avg']}")
        logging.info(f"평균 추론 시간: {stats['inference_ms']['avg']}ms")
        logging.info(f"평균 CPU 사용률: {stats['cpu']['avg']}%")
        logging.info(f"평균 RAM 사용률: {stats['ram']['avg']}%")
        logging.info(f"JSON 결과 저장: {json_path}")
        logging.info("===================")
        
        return stats


if __name__ == "__main__":
    logging.info(f"모델 경로: {MODEL_PATH}")
    logging.info(f"클래스 경로: {CLASSES_PATH}")
    logging.info(f"영상 소스: {SOURCE}")
    logging.info(f"측정 시간: {DURATION}초")
    
    analyzer = RealtimeInferenceAnalyzer(
        model_path=MODEL_PATH,
        classes_path=CLASSES_PATH,
        source=SOURCE,
        duration=DURATION,
        warmup_frames=WARMUP_FRAMES
    )
    results = analyzer.run()
    if results:
        logging.info("추론 분석 완료")
        logging.info(f"로그 파일 위치: {LOG_PATH}")