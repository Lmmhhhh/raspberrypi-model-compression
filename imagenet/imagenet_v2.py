import os
import io
import time
import torch
import csv
import traceback
import json
import urllib.request
from datetime import datetime
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# ===== 설정 =====
MODEL_NAME = "mobilenet_v3_small"  # 실행할 모델 이름만 바꿔서 재사용
DATA_DIR = "/home/minha/.cache/kagglehub/datasets/tusonggao/imagenet-validation-dataset/versions/1/imagenet_validation"
BATCH_SIZE = 16
DEVICE = torch.device("cpu")
OUTPUT_CSV = "/home/minha/raspberrypi/imagenet/eval_.csv"
LOG_FILE = "/home/minha/raspberrypi/imagenet/log_eval.txt"
DETAIL_LOG_FILE = "/home/minha/raspberrypi/imagenet/mobilenet_v3_small.log"  # 상세 예측 로그

# ===== 상세 로그 설정 =====
LOG_EVERY_N_IMAGES = 100  # N개마다 로그 (전체 로그는 너무 많을 수 있음)
LOG_FIRST_N_IMAGES = 50   # 처음 N개는 무조건 로그
LOG_ERRORS_ONLY = False   # True면 틀린 것만 로그

# ImageNet 클래스 라벨 다운로드 및 로드
import json
import urllib.request

def download_imagenet_labels():
    """ImageNet 라벨을 다운로드하거나 로컬 파일에서 로드"""
    labels_file = "/home/minha/raspberrypi/imagenet/imagenet_labels.json"
    
    # 이미 다운로드했으면 로드
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            return json.load(f)
    
    # 여러 소스 시도
    sources = [
        # Keras에서 사용하는 공식 매핑
        "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
        # 대체 소스
        "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    ]
    
    for url in sources:
        try:
            print(f"ImageNet 라벨 다운로드 중: {url}")
            response = urllib.request.urlopen(url)
            data = json.loads(response.read())
            
            # 파일 저장
            os.makedirs(os.path.dirname(labels_file), exist_ok=True)
            with open(labels_file, 'w') as f:
                json.dump(data, f)
            
            print(f"라벨 저장 완료: {labels_file}")
            return data
        except Exception as e:
            print(f"다운로드 실패: {e}")
    
    # 모든 소스 실패시 기본값
    print("ImageNet 라벨 다운로드 실패. 기본값 사용.")
    return None

# 전역 변수로 라벨 로드
IMAGENET_LABELS = download_imagenet_labels()

def get_imagenet_class_info(class_idx):
    """인덱스에서 synset ID와 human-readable 이름 반환"""
    if IMAGENET_LABELS and str(class_idx) in IMAGENET_LABELS:
        synset_id, class_name = IMAGENET_LABELS[str(class_idx)]
        return synset_id, class_name
    else:
        # 라벨이 없으면 기본값
        return f"n{class_idx:08d}", f"class_{class_idx}"

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
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 클래스 인덱스를 실제 ImageNet 클래스로 매핑
class_to_idx = dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

# ===== 평가 =====
try:
    with open(LOG_FILE, 'a') as logf, \
         open(OUTPUT_CSV, 'a', newline='') as csvfile, \
         open(DETAIL_LOG_FILE, 'a') as detail_log:
        
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["model", "top1_accuracy", "top5_accuracy", "avg_confidence", 
                           "avg_inference_time_ms", "model_size_mb"])
            logf.write(f"==== Started evaluation at {datetime.now().isoformat()} ====\n")
        
        logf.write(f"\n[{MODEL_NAME}] 시작: {datetime.now().isoformat()}\n")
        detail_log.write(f"\n{'='*80}\n")
        detail_log.write(f"[{MODEL_NAME}] 상세 예측 로그 - {datetime.now().isoformat()}\n")
        detail_log.write(f"{'='*80}\n\n")
        
        model = getattr(models, MODEL_NAME)(weights="DEFAULT").to(DEVICE)
        model.eval()
        
        top1_correct = 0
        top5_correct = 0
        total_images = 0
        confidences = []
        times = []
        
        # 이미지별 카운터
        image_counter = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=MODEL_NAME, unit="batch")):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                batch_size = images.size(0)
                
                start = time.perf_counter()
                outputs = model(images)
                end = time.perf_counter()
                
                times.append((end - start) * 1000 / batch_size)
                
                probs = torch.nn.functional.softmax(outputs, dim=1)
                max_probs, preds = probs.max(dim=1)
                confidences.extend(max_probs.tolist())
                
                # Top-5 예측 (logits와 indices)
                top5_logits, top5_preds = outputs.topk(5, dim=1)
                # Top-5에 대한 softmax 확률 계산
                top5_probs = torch.nn.functional.softmax(top5_logits, dim=1)
                
                # 배치 내 각 이미지에 대해
                for i in range(batch_size):
                    image_idx = batch_idx * BATCH_SIZE + i
                    true_label = labels[i].item()
                    pred_label = preds[i].item()
                    confidence = max_probs[i].item()
                    
                    # Top-1 정확도
                    is_correct = (pred_label == true_label)
                    if is_correct:
                        top1_correct += 1
                    
                    # Top-5 정확도
                    if true_label in top5_preds[i]:
                        top5_correct += 1
                    
                    # 상세 로그 조건 확인
                    should_log = False
                    if image_counter < LOG_FIRST_N_IMAGES:
                        should_log = True
                    elif image_counter % LOG_EVERY_N_IMAGES == 0:
                        should_log = True
                    elif LOG_ERRORS_ONLY and not is_correct:
                        should_log = True
                    
                    if should_log:
                        # 이미지 경로 추정 (ImageFolder 구조)
                        img_path = dataset.imgs[image_idx][0]
                        
                        # synset ID와 클래스 이름 가져오기
                        true_synset, true_name = get_imagenet_class_info(true_label)
                        pred_synset, pred_name = get_imagenet_class_info(pred_label)
                        
                        detail_log.write(f"[Image #{image_counter + 1}]\n")
                        detail_log.write(f"  파일: {img_path}\n")
                        detail_log.write(f"  정답: {true_synset} ({true_name}) [idx: {true_label}]\n")
                        detail_log.write(f"  예측: {pred_synset} ({pred_name}) [idx: {pred_label}]\n")
                        detail_log.write(f"  신뢰도: {confidence:.4f}\n")
                        detail_log.write(f"  정답여부: {'맞음' if is_correct else '틀림'}\n")
                        detail_log.write(f"  Top-5 예측:\n")
                        
                        for j in range(5):
                            top5_idx = top5_preds[i][j].item()
                            top5_prob = top5_probs[i][j].item()  # 이미 계산된 softmax 확률 사용
                            top5_synset, top5_name = get_imagenet_class_info(top5_idx)
                            detail_log.write(f"    {j+1}. {top5_synset} ({top5_name}) - {top5_prob:.4f}\n")
                        
                        detail_log.write(f"\n")
                        
                        # 콘솔에도 일부 출력 (선택적)
                        if image_counter < 10 or not is_correct:
                            print(f"\n[{image_counter + 1}] {os.path.basename(img_path)}: "
                                  f"예측={pred_name}, "
                                  f"정답={true_name}, "
                                  f"{'O' if is_correct else 'X'}")
                    
                    image_counter += 1
                    total_images += 1
        
        # 최종 통계
        top1_acc = top1_correct / total_images
        top5_acc = top5_correct / total_images
        avg_conf = sum(confidences) / len(confidences)
        avg_time = sum(times) / len(times)
        
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size_mb = buffer.getbuffer().nbytes / 1e6
        
        # CSV 저장
        writer.writerow([MODEL_NAME, f"{top1_acc:.4f}", f"{top5_acc:.4f}", 
                        f"{avg_conf:.4f}", f"{avg_time:.2f}", f"{size_mb:.2f}"])
        
        # 요약 로그
        summary = (f"[{MODEL_NAME}] 완료:\n"
                  f"  - Top1 정확도: {top1_acc:.4f} ({top1_correct}/{total_images})\n"
                  f"  - Top5 정확도: {top5_acc:.4f} ({top5_correct}/{total_images})\n"
                  f"  - 평균 신뢰도: {avg_conf:.4f}\n"
                  f"  - 평균 추론시간: {avg_time:.2f}ms\n"
                  f"  - 모델 크기: {size_mb:.2f}MB\n")
        
        logf.write(summary)
        detail_log.write(f"\n{'='*80}\n최종 결과:\n{summary}\n{'='*80}\n")
        
        print(f"\n{summary}")
        print(f"상세 예측 로그는 {DETAIL_LOG_FILE}에서 확인하세요.")

except Exception as e:
    with open(LOG_FILE, 'a') as logf:
        logf.write(f"[ERROR] [{MODEL_NAME}] 오류 발생 at {datetime.now().isoformat()}:\n")
        logf.write(traceback.format_exc())
    print(f"[ERROR] {e}")
    traceback.print_exc()