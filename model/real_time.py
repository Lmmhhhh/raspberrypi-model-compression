import cv2
import time
import urllib.request
import datetime

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import psutil

# ——— 설정 ———
STREAM_URL   =  "http://192.168.0.101:8081/video" #"http://192.168.219.232:8081/video" 
LABELS_URL   = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
LOG_FILENAME = "log_mobilenet_v2_.csv" #<-csv 파일명 수정정
MODEL_NAME   = "mobilenet_v2" #<-모델 수정 

# ——— 실험 시작 시각 기록 ———
exp_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ——— 모델 및 라벨 로드 ———
def load_model():
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT) #<-이 줄도 수정 
    m.eval()
    return m

def load_labels():
    return urllib.request.urlopen(LABELS_URL).read().decode().splitlines()

model  = load_model()
labels = load_labels()

# ——— 전처리 파이프라인 ———
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# ——— 비디오 스트림 열기 ———
cap = cv2.VideoCapture(STREAM_URL)

# ——— 실험용 변수 초기화 ———
frame_idx   = 0
start_time  = time.time()
fps_list    = []
infer_times = []
conf_list   = []
cpu_list    = []
ram_list    = []

# ——— 로그 파일 헤더 및 시작 시각 작성 ———
with open(LOG_FILENAME, "w") as log_f:
    log_f.write(f"experiment_start,{exp_start}\n")
    log_f.write("frame,label,fps,infer_time_ms,confidence,cpu_percent,ram_percent\n")

# ——— 메인 루프 ———
while True:
    ret, frame = cap.read()
    if not ret:
        print("스트림 프레임 수신 실패")
        break

    # 1) 전처리
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    input_t = preprocess(pil_img).unsqueeze(0)

    # 2) 추론 시간 측정
    t0       = time.perf_counter()
    with torch.no_grad():
        output = model(input_t)
    infer_ms = (time.perf_counter() - t0) * 1000
    infer_times.append(infer_ms)

    # 3) 예측 및 confidence
    pred       = torch.argmax(output, 1).item()
    label      = labels[pred]
    probs      = torch.nn.functional.softmax(output[0], dim=0)
    confidence = probs[pred].item()
    conf_list.append(confidence)

    # 4) FPS 계산
    frame_idx += 1
    elapsed    = time.time() - start_time
    fps        = frame_idx / elapsed
    fps_list.append(fps)

    # 5) 시스템 리소스 측정
    cpu_pct = psutil.cpu_percent()
    ram_pct = psutil.virtual_memory().percent
    cpu_list.append(cpu_pct)
    ram_list.append(ram_pct)

    # 6) 로그 남기기
    with open(LOG_FILENAME, "a") as log_f:
        log_f.write(f"{frame_idx},{label},{fps:.2f},{infer_ms:.2f},"
                    f"{confidence:.4f},{cpu_pct},{ram_pct}\n")

    # 7) 화면 표시
    h, w = frame.shape[:2]
    cv2.putText(frame, f"{label} ({confidence:.2f})", (w-350, h-90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, f"FPS:{fps:.1f}  Time:{infer_ms:.1f}ms", (w-350, h-60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"CPU:{cpu_pct:.0f}% RAM:{ram_pct:.0f}%", (w-350, h-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.imshow(f"{MODEL_NAME} - Real-Time", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

# ——— 실험 종료 시각 기록 ———
exp_end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(LOG_FILENAME, "a") as log_f:
    log_f.write(f"experiment_end,{exp_end}\n")

# ——— 요약 계산 및 출력/기록 ———
if fps_list:
    avg_fps   = sum(fps_list) / len(fps_list)
    avg_ms    = sum(infer_times) / len(infer_times)
    avg_conf  = sum(conf_list) / len(conf_list)
    avg_cpu   = sum(cpu_list) / len(cpu_list)
    avg_ram   = sum(ram_list) / len(ram_list)

    print(f"[{MODEL_NAME}] 평균 FPS: {avg_fps:.2f}, 평균 추론시간: {avg_ms:.1f} ms, "
          f"평균 Confidence: {avg_conf:.3f}, CPU: {avg_cpu:.1f}%, RAM: {avg_ram:.1f}%")

    with open(LOG_FILENAME, "a") as log_f:
        log_f.write(f"average,,{avg_fps:.2f},{avg_ms:.1f},{avg_conf:.3f},"
                    f"{avg_cpu:.1f},{avg_ram:.1f}\n") 