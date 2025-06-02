import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import time
from PIL import Image

# 모델 로딩
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.eval()

# 전처리 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# 카메라 스트림
cap = cv2.VideoCapture("http://192.168.0.101:8081/video")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

frame_count = 0
start_time = time.time()

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 입력 실패")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        input_tensor = transform(pil_img).unsqueeze(0)

        # 추론
        t0 = time.time()
        output = model(input_tensor)
        t1 = time.time()

        fps = 1 / (t1 - t0)
        print(f"Inference time: {(t1 - t0)*1000:.1f} ms | FPS: {fps:.1f}")

        # ESC 종료
        cv2.imshow("MobileNetV2", frame)
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()