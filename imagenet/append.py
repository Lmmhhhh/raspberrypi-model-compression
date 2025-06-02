import csv
import torch
from torchvision import models
from ptflops import get_model_complexity_info
import traceback
from datetime import datetime

# CSV 파일 경로
INPUT_CSV = "/home/minha/raspberrypi/imagenet/eval.csv"
OUTPUT_CSV = "/home/minha/raspberrypi/imagenet/eval_with_flops.csv"
LOG_FILE = "/home/minha/raspberrypi/imagenet/flops_calculation.log"

# 디바이스 설정
DEVICE = torch.device("cpu")

def log_message(message, log_file=LOG_FILE):
    """로그 파일에 메시지 기록"""
    with open(log_file, 'a') as f:
        timestamp = datetime.now().isoformat()
        f.write(f"[{timestamp}] {message}\n")
    # 콘솔에도 출력 (선택사항)
    print(message)

def get_model_complexity(model_name):
    """모델의 FLOPs와 파라미터 수 계산"""
    try:
        log_message(f"\n처리 중: {model_name}")
        
        # 모델 로드
        model = getattr(models, model_name)(weights="DEFAULT").to(DEVICE)
        model.eval()
        
        # FLOPs와 파라미터 계산
        with torch.no_grad():
            macs, params = get_model_complexity_info(
                model, 
                (3, 224, 224), 
                as_strings=False, 
                print_per_layer_stat=False,
                verbose=False
            )
            
        gflops = macs / 1e9
        params_m = params / 1e6
        
        log_message(f"  - GFLOPs: {gflops:.2f}")
        log_message(f"  - Parameters: {params_m:.2f}M")
        
        return gflops, params_m
        
    except Exception as e:
        log_message(f"  - Error: {e}")
        log_message(traceback.format_exc())
        return None, None

def batch_update_csv():
    """모든 모델에 대해 FLOPs와 파라미터 추가"""
    
    log_message("\n" + "="*60)
    log_message("일괄 FLOPs/Parameters 계산 시작")
    log_message("="*60)
    
    # 1. 기존 CSV 읽기
    rows = []
    model_names = []
    
    with open(INPUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        
        for row in reader:
            rows.append(row)
            model_names.append(row['model'])
    
    # 2. 헤더에 새 열 추가
    if 'gflops' not in fieldnames:
        fieldnames.append('gflops')
    if 'params_m' not in fieldnames:
        fieldnames.append('params_m')
    
    log_message(f"\n발견된 모델 수: {len(model_names)}")
    log_message(f"모델 목록: {', '.join(model_names)}")
    
    # 3. 각 모델에 대해 복잡도 계산
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for i, row in enumerate(rows):
        model_name = row['model']
        
        # 이미 계산된 경우 스킵
        if 'gflops' in row and row.get('gflops') and row.get('gflops') != 'N/A':
            log_message(f"\n{model_name}: 이미 계산됨 (GFLOPs: {row['gflops']}, Params: {row['params_m']}M) - 스킵")
            skip_count += 1
            continue
        
        gflops, params_m = get_model_complexity(model_name)
        
        if gflops is not None:
            row['gflops'] = f"{gflops:.2f}"
            row['params_m'] = f"{params_m:.2f}"
            success_count += 1
            log_message(f"  - 성공적으로 계산 완료")
        else:
            row['gflops'] = 'N/A'
            row['params_m'] = 'N/A'
            fail_count += 1
            log_message(f"  - 계산 실패 (N/A로 표시)")
    
    # 4. 업데이트된 CSV 저장
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    log_message(f"\n\n=== 처리 완료 ===")
    log_message(f"성공: {success_count}개")
    log_message(f"실패: {fail_count}개")
    log_message(f"스킵: {skip_count}개")
    log_message(f"결과 저장: {OUTPUT_CSV}")
    
    # 5. 최종 결과 테이블 로그
    log_message("\n최종 결과:")
    log_message("-" * 130)
    header = f"{'Model':<20} {'Top1':<6} {'Top5':<6} {'Conf':<6} {'Time(ms)':<9} {'Size(MB)':<9} {'GFLOPs':<8} {'Params(M)':<10}"
    log_message(header)
    log_message("-" * 130)
    
    for row in rows:
        row_str = (f"{row['model']:<20} "
                  f"{float(row['top1_accuracy']):<6.4f} "
                  f"{float(row['top5_accuracy']):<6.4f} "
                  f"{float(row['avg_confidence']):<6.4f} "
                  f"{float(row['avg_inference_time_ms']):<9.2f} "
                  f"{float(row['model_size_mb']):<9.2f} "
                  f"{row.get('gflops', 'N/A'):<8} "
                  f"{row.get('params_m', 'N/A'):<10}")
        log_message(row_str)
    
    log_message("-" * 130)
    
    log_message("\n" + "="*60)
    log_message("일괄 처리 완료")
    log_message("="*60 + "\n")

# 실행
if __name__ == "__main__":
    batch_update_csv()