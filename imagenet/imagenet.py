import os
import io
import time
import torch
import csv
import traceback
from datetime import datetime
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# ===== 설정 =====
MODEL_NAME  = "squeezenet1_1"  # 실행할 모델 이름만 바꿔서 재사용
DATA_DIR    = "/home/minha/.cache/kagglehub/datasets/tusonggao/imagenet-validation-dataset/versions/1/imagenet_validation"
BATCH_SIZE  = 16
DEVICE      = torch.device("cpu")
OUTPUT_CSV  = "/home/minha/raspberrypi/imagenet/eval.csv"
LOG_FILE    = "/home/minha/raspberrypi/imagenet/log_eval.txt"

# ===== CSV 헤더가 없는 경우만 작성 =====
write_header = not os.path.exists(OUTPUT_CSV)

# ===== 데이터 로더 준비 =====
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ===== 평가 =====
try:
    with open(LOG_FILE, 'a') as logf, open(OUTPUT_CSV, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["model", "top1_accuracy", "top5_accuracy", "avg_confidence", "avg_inference_time_ms", "model_size_mb"])
            logf.write(f"==== Started evaluation at {datetime.now().isoformat()} ====\n")

        logf.write(f"\n▶️ [{MODEL_NAME}] 시작: {datetime.now().isoformat()}\n")

        model = getattr(models, MODEL_NAME)(weights="DEFAULT").to(DEVICE)
        model.eval()

        top1_correct = 0
        top5_correct = 0
        total_images = 0
        confidences = []
        times = []

        with torch.no_grad():
            for images, labels in tqdm(loader, desc=MODEL_NAME, unit="batch"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                batch_size = images.size(0)

                start = time.perf_counter()
                outputs = model(images)
                end = time.perf_counter()

                times.append((end - start) * 1000 / batch_size)

                probs = torch.nn.functional.softmax(outputs, dim=1)
                max_probs, preds = probs.max(dim=1)
                confidences.extend(max_probs.tolist())

                top1_correct += (preds == labels).sum().item()

                _, top5_preds = outputs.topk(5, dim=1)
                for i in range(batch_size):
                    if labels[i] in top5_preds[i]:
                        top5_correct += 1

                total_images += batch_size

        top1_acc = top1_correct / total_images
        top5_acc = top5_correct / total_images
        avg_conf = sum(confidences) / len(confidences)
        avg_time = sum(times) / len(times)

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size_mb = buffer.getbuffer().nbytes / 1e6

        writer.writerow([MODEL_NAME, f"{top1_acc:.4f}", f"{top5_acc:.4f}", f"{avg_conf:.4f}", f"{avg_time:.2f}", f"{size_mb:.2f}"])
        logf.write(f"[{MODEL_NAME}] 완료: Top1={top1_acc:.4f}, Top5={top5_acc:.4f}, Conf={avg_conf:.4f}, Time={avg_time:.2f}ms, Size={size_mb:.2f}MB\n")

except Exception as e:
    with open(LOG_FILE, 'a') as logf:
        logf.write(f"[{MODEL_NAME}] 오류 발생 at {datetime.now().isoformat()}:\n")
        logf.write(traceback.format_exc())
    print(f"[ERROR] {e}")

