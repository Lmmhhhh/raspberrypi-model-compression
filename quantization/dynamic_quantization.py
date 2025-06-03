"""
MobileNet 양자화 구현
Dynamic Quantization을 기본으로 사용
테스트 배치 포함된 버전 
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import sys
import json

class QuantizationLogger:
    """양자화 전용 로거"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'quantization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        self.logger = logging.getLogger('QuantizationLogger')
        self.logger.setLevel(logging.DEBUG)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("="*60)
        self.logger.info("양자화 로거 초기화")
        self.logger.info(f"로그 디렉토리: {log_dir}")
        self.logger.info("="*60)
    
    def info(self, msg):
        self.logger.info(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def debug(self, msg):
        self.logger.debug(msg)
    
    def save_checkpoint(self, data, filename):
        """중간 결과 저장"""
        try:
            filepath = os.path.join(self.log_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            self.info(f"체크포인트 저장: {filepath}")
        except Exception as e:
            self.error(f"체크포인트 저장 실패: {str(e)}")

class MobileNetQuantization:
    """MobileNet 양자화 클래스 - 수정된 버전"""
    
    def __init__(self, model_name='mobilenet_v3_small', device='cpu', base_dir='quantization_results'):
        self.model_name = model_name
        self.device = device
        
        self.log_dir = os.path.join(base_dir, f'{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.logger = QuantizationLogger(self.log_dir)
        self.logger.info(f"MobileNetQuantization 초기화: {model_name}")
        
        # PyTorch 버전 확인
        self.logger.info(f"PyTorch 버전: {torch.__version__}")
        
        try:
            self.model = self._load_model()
            self.logger.info("모델 로드 성공")
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {str(e)}")
            raise
    
    def _load_model(self):
        """사전학습된 모델 로드"""
        self.logger.info(f"모델 로드 중: {self.model_name}")
        
        if self.model_name == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        elif self.model_name == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(weights='IMAGENET1K_V2')
        elif self.model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights='IMAGENET1K_V2')
        else:
            raise ValueError(f"지원하지 않는 모델: {self.model_name}")
        
        self.logger.info(f"모델 아키텍처: {self.model_name}")
        self.logger.info(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def prepare_data_loaders(self, train_path, val_path, batch_size=32, num_workers=4):
        """데이터 로더 준비"""
        self.logger.info("="*60)
        self.logger.info("데이터 로더 준비 시작")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"학습 데이터 경로가 존재하지 않습니다: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"검증 데이터 경로가 존재하지 않습니다: {val_path}")
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = datasets.ImageFolder(train_path, transform=transform)
        val_dataset = datasets.ImageFolder(val_path, transform=transform)
        
        self.logger.info(f"학습 데이터: {len(train_dataset)} 샘플")
        self.logger.info(f"검증 데이터: {len(val_dataset)} 샘플")
        self.logger.info(f"클래스 수: {len(train_dataset.classes)}")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader
    
    def evaluate_model_safe(self, model, val_loader, desc="평가", max_batches=None):
        """안전한 모델 평가 - 오류 처리 강화"""
        self.logger.info(f"{desc} 시작...")
        model.eval()
        correct = 0
        total = 0
        successful_batches = 0
        failed_batches = 0
        
        # 평가할 배치 수 제한 (테스트용)
        total_batches = len(val_loader) if max_batches is None else min(max_batches, len(val_loader))
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, desc=desc, total=total_batches)):
                if max_batches and batch_idx >= max_batches:
                    break
                    
                try:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = model(inputs)
                    
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    successful_batches += 1
                    
                except Exception as e:
                    failed_batches += 1
                    if failed_batches <= 5:  # 처음 5개 오류만 로깅
                        self.logger.warning(f"배치 {batch_idx} 평가 실패: {str(e)[:100]}")
                    continue
        
        if total > 0:
            accuracy = 100. * correct / total
        else:
            accuracy = 0.0
            
        self.logger.info(f"{desc} 완료 - 정확도: {accuracy:.2f}%")
        self.logger.info(f"성공한 배치: {successful_batches}, 실패한 배치: {failed_batches}")
        
        return accuracy
    
    def measure_inference_time(self, model, input_shape=(1, 3, 224, 224), num_runs=10):
        """추론 시간 측정 - 실행 횟수 줄임"""
        self.logger.info("추론 시간 측정 시작...")
        model.eval()
        
        try:
            dummy_input = torch.randn(input_shape).to(self.device)
            
            # Warm-up
            for _ in range(3):
                _ = model(dummy_input)
            
            # 실제 측정
            start_time = time.time()
            for _ in range(num_runs):
                _ = model(dummy_input)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs * 1000  # ms
            self.logger.info(f"평균 추론 시간: {avg_time:.2f} ms")
            return avg_time
            
        except Exception as e:
            self.logger.error(f"추론 시간 측정 실패: {str(e)}")
            return -1.0
    
    def get_model_size(self, model):
        """모델 크기 계산 (MB)"""
        try:
            # 모델을 임시 파일로 저장하여 크기 측정
            temp_path = os.path.join(self.log_dir, 'temp_model.pth')
            torch.save(model.state_dict(), temp_path)
            size_mb = os.path.getsize(temp_path) / 1024 / 1024
            os.remove(temp_path)
            
            self.logger.info(f"모델 크기: {size_mb:.2f} MB")
            return size_mb
            
        except Exception as e:
            self.logger.error(f"모델 크기 계산 실패: {str(e)}")
            return -1.0
    
    def quantize_model_dynamic(self, train_loader, val_loader, test_batches=100):
        """Dynamic Quantization 수행"""
        print("\n" + "="*60)
        print("Dynamic Quantization 시작")
        print("="*60)
        
        # 1. 원본 모델 평가
        print("\n1. 원본 모델 평가 (일부 배치만 사용)")
        original_accuracy = self.evaluate_model_safe(
            self.model, val_loader, "원본 모델 평가", max_batches=test_batches
        )
        original_size = self.get_model_size(self.model)
        original_time = self.measure_inference_time(self.model)
        
        print(f"원본 모델 정확도: {original_accuracy:.2f}%")
        print(f"원본 모델 크기: {original_size:.2f} MB")
        print(f"원본 추론 시간: {original_time:.2f} ms")
        
        # 2. Dynamic Quantization 수행
        print("\n2. Dynamic Quantization 수행")
        self.model.eval()
        self.model = self.model.cpu()
        self.device = 'cpu'
        
        # Dynamic Quantization
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv2d},  # 양자화할 레이어 타입
            dtype=torch.qint8
        )
        
        print("Dynamic Quantization 완료!")
        
        # 3. 양자화된 모델 평가
        print("\n3. 양자화된 모델 평가 (일부 배치만 사용)")
        quantized_accuracy = self.evaluate_model_safe(
            quantized_model, val_loader, "양자화 모델 평가", max_batches=test_batches
        )
        quantized_size = self.get_model_size(quantized_model)
        quantized_time = self.measure_inference_time(quantized_model)
        
        print(f"양자화 모델 정확도: {quantized_accuracy:.2f}%")
        print(f"양자화 모델 크기: {quantized_size:.2f} MB")
        print(f"양자화 추론 시간: {quantized_time:.2f} ms")
        
        # 4. 결과 비교
        print("\n4. 결과 비교")
        print(f"{'Model Type':<25} {'Accuracy (%)':<15} {'Size (MB)':<15} {'Time (ms)':<15}")
        print("-" * 70)
        print(f"{'Original (FP32)':<25} {original_accuracy:<15.2f} {original_size:<15.2f} {original_time:<15.2f}")
        print(f"{'Quantized (INT8)':<25} {quantized_accuracy:<15.2f} {quantized_size:<15.2f} {quantized_time:<15.2f}")
        print("-" * 70)
        
        if original_accuracy > 0:
            print(f"정확도 변화: {quantized_accuracy - original_accuracy:+.2f}%")
        if original_size > 0:
            print(f"모델 크기 감소: {(1 - quantized_size/original_size)*100:.1f}%")
        if original_time > 0 and quantized_time > 0:
            print(f"추론 속도 향상: {(original_time/quantized_time):.2f}x")
        
        # 5. 결과 저장
        results = {
            'quantization_type': 'dynamic',
            'original': {
                'accuracy': original_accuracy,
                'size_mb': original_size,
                'inference_ms': original_time
            },
            'quantized': {
                'accuracy': quantized_accuracy,
                'size_mb': quantized_size,
                'inference_ms': quantized_time
            },
            'test_batches': test_batches
        }
        
        self._save_results(quantized_model, results)
        
        return quantized_model
    
    def _save_results(self, quantized_model, results):
        """결과 저장"""
        self.logger.info("\n=== 결과 저장 중 ===")
        
        try:
            # 1. 양자화된 모델 저장
            model_path = os.path.join(self.log_dir, f'{self.model_name}_quantized_dynamic.pth')
            torch.save(quantized_model, model_path)
            self.logger.info(f"양자화 모델 저장: {model_path}")
            
            # 2. 결과 JSON 저장
            results_path = os.path.join(self.log_dir, 'quantization_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"결과 데이터 저장: {results_path}")
            
            # 3. 요약 텍스트 저장
            summary_path = os.path.join(self.log_dir, 'summary.txt')
            with open(summary_path, 'w') as f:
                f.write("Dynamic Quantization 결과 요약\n")
                f.write("="*60 + "\n")
                f.write(f"모델: {self.model_name}\n")
                f.write(f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"양자화 방식: Dynamic Quantization\n\n")
                
                f.write("원본 모델:\n")
                f.write(f"  정확도: {results['original']['accuracy']:.2f}%\n")
                f.write(f"  크기: {results['original']['size_mb']:.2f} MB\n")
                f.write(f"  추론시간: {results['original']['inference_ms']:.2f} ms\n\n")
                
                f.write("양자화 모델:\n")
                f.write(f"  정확도: {results['quantized']['accuracy']:.2f}%\n")
                f.write(f"  크기: {results['quantized']['size_mb']:.2f} MB\n")
                f.write(f"  추론시간: {results['quantized']['inference_ms']:.2f} ms\n\n")
                
                if results['original']['accuracy'] > 0 and results['original']['size_mb'] > 0:
                    f.write("개선 효과:\n")
                    f.write(f"  정확도 변화: {results['quantized']['accuracy'] - results['original']['accuracy']:+.2f}%\n")
                    f.write(f"  크기 감소: {(1 - results['quantized']['size_mb']/results['original']['size_mb'])*100:.1f}%\n")
                    if results['original']['inference_ms'] > 0 and results['quantized']['inference_ms'] > 0:
                        f.write(f"  속도 향상: {results['original']['inference_ms']/results['quantized']['inference_ms']:.2f}x\n")
            
            self.logger.info(f"요약 저장: {summary_path}")
            
            # 4. 간단한 시각화
            try:
                self._plot_simple_results(results)
            except Exception as e:
                self.logger.warning(f"그래프 생성 실패: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"결과 저장 중 오류: {str(e)}")
    
    def _plot_simple_results(self, results):
        """간단한 결과 시각화"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        metrics = ['Accuracy (%)', 'Size (MB)', 'Time (ms)']
        original_values = [
            results['original']['accuracy'],
            results['original']['size_mb'],
            results['original']['inference_ms']
        ]
        quantized_values = [
            results['quantized']['accuracy'],
            results['quantized']['size_mb'],
            results['quantized']['inference_ms']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, original_values, width, label='Original', color='blue')
        bars2 = ax.bar(x + width/2, quantized_values, width, label='Quantized', color='green')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Quantization Results Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # 값 표시
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.1f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'quantization_results.png'), dpi=150)
        plt.close()
        
        self.logger.info("결과 그래프 저장 완료")


# 메인 실행 코드
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MobileNet Dynamic Quantization')
    parser.add_argument('--model', type=str, default='mobilenet_v3_small',
                        choices=['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'])
    parser.add_argument('--train_data', type=str, required=True,
                        help='학습 데이터 경로')
    parser.add_argument('--val_data', type=str, required=True,
                        help='검증 데이터 경로')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='배치 크기')
    parser.add_argument('--test_batches', type=int, default=100,
                        help='평가에 사용할 배치 수 (빠른 테스트용)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MobileNet Dynamic Quantization")
    print("="*60)
    print(f"PyTorch 버전: {torch.__version__}")
    print("="*60)
    
    try:
        # 1. 양자화 객체 생성
        quantizer = MobileNetQuantization(model_name=args.model)
        
        # 2. 데이터 로더 준비
        print("\n데이터 로더 준비 중...")
        train_loader, val_loader = quantizer.prepare_data_loaders(
            args.train_data, args.val_data, args.batch_size
        )
        
        # 3. Dynamic Quantization 실행
        quantized_model = quantizer.quantize_model_dynamic(
            train_loader, val_loader, 
            test_batches=args.test_batches
        )
        
        # 4. 완료
        print("\n" + "="*60)
        print("Dynamic Quantization 완료!")
        print("="*60)
        print(f"\n생성된 파일 위치: {quantizer.log_dir}")
        print("\n생성된 파일:")
        print(f"  - {args.model}_quantized_dynamic.pth")
        print("  - quantization_results.json")
        print("  - summary.txt")
        print("  - quantization_results.png")
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()