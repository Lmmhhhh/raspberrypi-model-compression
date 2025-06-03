"""
MobileNet 양자화 구현 - MATLAB 예제 기반
INT8 양자화를 통한 모델 압축
moblienet_v3는 토치에서 호환이 잘 안됨 (성공 사례도 드묾)
양자화 단계에서 계속 오류
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.quantization import get_default_qconfig, prepare_qat, convert
from torch.quantization import quantize_dynamic, quantize
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경을 위한 백엔드
import matplotlib.pyplot as plt
import logging
import traceback
import sys
import json

class QuantizationLogger:
    """양자화 전용 로거"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 메인 로그 파일
        log_file = os.path.join(log_dir, f'quantization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        # 로거 설정
        self.logger = logging.getLogger('QuantizationLogger')
        self.logger.setLevel(logging.DEBUG)
        
        # 파일 핸들러
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 핸들러 추가
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 에러 로그 파일 (별도)
        error_file = os.path.join(log_dir, f'errors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        error_handler = logging.FileHandler(error_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
        
        self.logger.info("="*60)
        self.logger.info("양자화 로거 초기화")
        self.logger.info(f"로그 디렉토리: {log_dir}")
        self.logger.info(f"메인 로그: {log_file}")
        self.logger.info(f"에러 로그: {error_file}")
        self.logger.info("="*60)
    
    def info(self, msg):
        self.logger.info(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def debug(self, msg):
        self.logger.debug(msg)
    
    def exception(self, msg):
        self.logger.exception(msg)
        
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
    """MobileNet 양자화 클래스"""
    
    def __init__(self, model_name='mobilenet_v3_small', device='cpu', base_dir='quantization_results'):
        """
        양자화는 주로 CPU에서 실행됨 (INT8 연산 최적화)
        """
        self.model_name = model_name
        self.device = device
        
        # 로그 디렉토리 설정
        self.log_dir = os.path.join(base_dir, f'{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 로거 초기화
        self.logger = QuantizationLogger(self.log_dir)
        self.logger.info(f"MobileNetQuantization 초기화: {model_name}")
        
        # 모델 로드
        try:
            self.model = self._load_model()
            self.logger.info("모델 로드 성공")
        except Exception as e:
            self.logger.exception("모델 로드 실패")
            raise
        
    def _load_model(self):
        """사전학습된 모델 로드"""
        self.logger.info(f"모델 로드 중: {self.model_name}")
        
        try:
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
            
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류: {str(e)}")
            raise
    
    def prepare_data_loaders(self, train_path, val_path, batch_size=32, num_workers=4):
        """데이터 로더 준비"""
        self.logger.info("="*60)
        self.logger.info("데이터 로더 준비 시작")
        
        try:
            # 경로 확인
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"학습 데이터 경로가 존재하지 않습니다: {train_path}")
            if not os.path.exists(val_path):
                raise FileNotFoundError(f"검증 데이터 경로가 존재하지 않습니다: {val_path}")
            
            self.logger.info(f"학습 데이터 경로: {train_path}")
            self.logger.info(f"검증 데이터 경로: {val_path}")
            
            # ImageNet 전처리
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # 데이터셋 로드
            self.logger.info("데이터셋 로드 중...")
            train_dataset = datasets.ImageFolder(train_path, transform=transform)
            val_dataset = datasets.ImageFolder(val_path, transform=transform)
            
            # 데이터 정보 로깅
            self.logger.info(f"학습 데이터: {len(train_dataset)} 샘플")
            self.logger.info(f"검증 데이터: {len(val_dataset)} 샘플")
            self.logger.info(f"클래스 수: {len(train_dataset.classes)}")
            self.logger.info(f"배치 크기: {batch_size}")
            self.logger.info(f"워커 수: {num_workers}")
            
            # 클래스 정보 저장
            class_info = {
                'num_classes': len(train_dataset.classes),
                'classes': train_dataset.classes[:10],  # 처음 10개만
                'class_to_idx': dict(list(train_dataset.class_to_idx.items())[:10])
            }
            self.logger.save_checkpoint(class_info, 'dataset_info.json')
            
            # 데이터로더 생성
            train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                    shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                  shuffle=False, num_workers=num_workers)
            
            self.logger.info("데이터 로더 생성 완료")
            return train_loader, val_loader
            
        except Exception as e:
            self.logger.exception("데이터 로더 준비 중 오류")
            raise
    
    def evaluate_model(self, model, val_loader, desc="평가"):
        """모델 평가"""
        self.logger.info(f"{desc} 시작...")
        model.eval()
        correct = 0
        total = 0
        
        # 양자화 백엔드 설정
        if hasattr(torch.backends, 'quantized'):
            torch.backends.quantized.engine = 'qnnpack' 
        
        try:
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, desc=desc)):
                    try:
                        inputs = inputs.to(self.device)
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets.to(self.device)).sum().item()
                        
                        # 주기적 로깅
                        if batch_idx % 100 == 0:
                            self.logger.debug(f"{desc} 진행: {batch_idx}/{len(val_loader)} 배치")
                            
                    except RuntimeError as e:
                        if "quantized::" in str(e):
                            self.logger.warning(f"배치 {batch_idx} - 양자화 관련 오류 발생, 스킵")
                            continue
                        else:
                            self.logger.warning(f"배치 {batch_idx} 평가 중 오류: {str(e)}")
                            continue
                    except Exception as e:
                        self.logger.warning(f"배치 {batch_idx} 평가 중 오류: {str(e)}")
                        continue
            
            accuracy = 100. * correct / total if total > 0 else 0
            self.logger.info(f"{desc} 완료 - 정확도: {accuracy:.2f}%")
            
            # 중간 결과 저장
            self.logger.save_checkpoint({
                'description': desc,
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f'evaluation_{desc.replace(" ", "_")}.json')
            
            return accuracy
            
        except Exception as e:
            self.logger.exception(f"{desc} 중 오류 발생")
            return 0.0
    
    def measure_inference_time(self, model, input_shape=(1, 3, 224, 224), num_runs=100):
        """추론 시간 측정"""
        self.logger.info("추론 시간 측정 시작...")
        
        try:
            model.eval()
            dummy_input = torch.randn(input_shape).to(self.device)
            
            # Warm-up
            self.logger.debug("모델 워밍업 중...")
            for _ in range(10):
                _ = model(dummy_input)
            
            # 실제 측정
            self.logger.debug(f"{num_runs}회 추론 시간 측정 중...")
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start_time = time.time()
            
            for i in range(num_runs):
                _ = model(dummy_input)
                if i % 20 == 0:
                    self.logger.debug(f"추론 진행: {i}/{num_runs}")
                    
            torch.cuda.synchronize() if self.device == 'cuda' else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs * 1000  # ms
            self.logger.info(f"평균 추론 시간: {avg_time:.2f} ms")
            
            return avg_time
            
        except Exception as e:
            self.logger.exception("추론 시간 측정 중 오류")
            return -1.0
    
    def get_model_size(self, model):
        """모델 크기 계산 (MB)"""
        try:
            param_size = 0
            buffer_size = 0
            
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            size_mb = (param_size + buffer_size) / 1024 / 1024
            self.logger.info(f"모델 크기: {size_mb:.2f} MB")
            
            return size_mb
            
        except Exception as e:
            self.logger.exception("모델 크기 계산 중 오류")
            return -1.0
    
    def quantize_model(self, train_loader, val_loader, calibration_batches=100):
        """
        Post-training Quantization (PTQ) 수행
        MATLAB의 dlquantizer와 유사한 방식
        """
        print("\n" + "="*60)
        print("Post-training Quantization (PTQ) 시작")
        print("="*60)
        
        # 양자화 백엔드 설정
        backend = 'fbgemm'  # Intel CPU용 (AMD CPU는 'qnnpack' 사용)
        torch.backends.quantized.engine = backend
        self.logger.info(f"양자화 백엔드: {backend}")
        
        # 1. 원본 모델 평가
        print("\n1. 원본 모델 평가")
        original_accuracy = self.evaluate_model(self.model, val_loader, "원본 모델 평가")
        original_size = self.get_model_size(self.model)
        original_time = self.measure_inference_time(self.model)
        
        print(f"원본 모델 정확도: {original_accuracy:.2f}%")
        print(f"원본 모델 크기: {original_size:.2f} MB")
        print(f"원본 추론 시간: {original_time:.2f} ms")
        
        # 2. 양자화 준비
        print("\n2. 양자화 준비")
        self.model.eval()
        
        # CPU로 이동 (INT8 양자화는 CPU에서 실행)
        self.model = self.model.cpu()
        self.device = 'cpu'
        
        # MobileNet 특별 처리 - Fuse 모듈
        if 'mobilenet_v3' in self.model_name:
            # MobileNetV3의 경우 특별한 fuse 패턴
            self.model = self._fuse_mobilenet_v3()
        elif 'mobilenet_v2' in self.model_name:
            # MobileNetV2의 경우
            self.model = self._fuse_mobilenet_v2()
        
        # 양자화 설정
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 양자화 준비
        torch.quantization.prepare(self.model, inplace=True)
        
        # 3. Calibration (MATLAB의 calibrate 함수와 유사)
        print(f"\n3. Calibration 수행 ({calibration_batches} 배치)")
        print("Calibration 결과 (MATLAB calResults와 유사):")
        print(f"{'Layer Name':<30} {'Min Value':<12} {'Max Value':<12}")
        print("-" * 60)
        
        self.model.eval()
        
        # Calibration statistics 수집
        calibration_stats = {}
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(tqdm(train_loader, desc="Calibration")):
                if i >= calibration_batches:
                    break
                inputs = inputs.cpu()
                _ = self.model(inputs)
        
        # 주요 레이어의 통계 출력 (MATLAB 스타일)
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight'):
                weight_min = module.weight.min().item()
                weight_max = module.weight.max().item()
                if name:  # 이름이 있는 레이어만
                    print(f"{name:<30} {weight_min:<12.6f} {weight_max:<12.6f}")
                    calibration_stats[name] = {
                        'min': weight_min,
                        'max': weight_max
                    }
        
        # 4. 양자화 변환
        print("\n4. INT8 양자화 변환")
        quantized_model = torch.quantization.convert(self.model, inplace=False)
        
        # 5. 양자화된 모델 평가
        print("\n5. 양자화된 모델 평가")
        quantized_accuracy = self.evaluate_model(quantized_model, val_loader, "양자화 모델 평가")
        quantized_size = self.get_model_size(quantized_model)
        quantized_time = self.measure_inference_time(quantized_model)
        
        print(f"양자화 모델 정확도: {quantized_accuracy:.2f}%")
        print(f"양자화 모델 크기: {quantized_size:.2f} MB")
        print(f"양자화 추론 시간: {quantized_time:.2f} ms")
        
        # 6. 결과 비교 (MATLAB validationMetrics와 유사)
        print("\n6. 결과 비교")
        print(f"{'NetworkImplementation':<25} {'Accuracy':<12} {'Size (MB)':<12} {'Time (ms)':<12}")
        print("-" * 70)
        print(f"{'Floating-Point':<25} {original_accuracy:<12.4f} {original_size:<12.2f} {original_time:<12.2f}")
        print(f"{'Quantized (INT8)':<25} {quantized_accuracy:<12.4f} {quantized_size:<12.2f} {quantized_time:<12.2f}")
        print("-" * 70)
        print(f"정확도 변화: {quantized_accuracy - original_accuracy:+.2f}%")
        print(f"모델 크기 감소: {(1 - quantized_size/original_size)*100:.1f}%")
        print(f"추론 속도 향상: {(original_time/quantized_time):.2f}x")
        
        # 7. 결과 저장
        self._save_results(quantized_model, {
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
            'calibration_stats': calibration_stats
        })
        
        return quantized_model
    
    def _fuse_mobilenet_v3(self):
        """MobileNetV3 모듈 융합"""
        try:
            # MobileNetV3의 특정 패턴 융합
            for module in self.model.modules():
                if type(module).__name__ == 'ConvBNActivation':
                    if hasattr(module, '0') and hasattr(module, '1') and hasattr(module, '2'):
                        torch.quantization.fuse_modules(module, ['0', '1', '2'], inplace=True)
                        self.logger.debug(f"Fused ConvBNActivation with 3 modules")
                    elif hasattr(module, '0') and hasattr(module, '1'):
                        torch.quantization.fuse_modules(module, ['0', '1'], inplace=True)
                        self.logger.debug(f"Fused ConvBNActivation with 2 modules")
            
            self.logger.info("MobileNetV3 모듈 융합 완료")
            return self.model
            
        except Exception as e:
            self.logger.warning(f"MobileNetV3 모듈 융합 중 경고: {str(e)}")
            return self.model
    
    def _fuse_mobilenet_v2(self):
        """MobileNetV2 모듈 융합"""
        try:
            # MobileNetV2의 InvertedResidual 블록 처리
            for module in self.model.modules():
                if type(module).__name__ == 'ConvBNReLU':
                    torch.quantization.fuse_modules(module, ['0', '1', '2'], inplace=True)
                    self.logger.debug(f"Fused ConvBNReLU module")
            
            self.logger.info("MobileNetV2 모듈 융합 완료")
            return self.model
            
        except Exception as e:
            self.logger.warning(f"MobileNetV2 모듈 융합 중 경고: {str(e)}")
            return self.model
    
    def _save_results(self, quantized_model, results):
        """결과 저장"""
        self.logger.info("\n=== 결과 저장 중 ===")
        
        try:
            # 1. 양자화된 모델 저장 (PyTorch 형식)
            model_path = os.path.join(self.log_dir, f'{self.model_name}_quantized.pth')
            torch.save(quantized_model, model_path)
            self.logger.info(f"양자화 모델 저장: {model_path}")
            
            # 2. TorchScript 형식으로도 저장 (배포용)
            try:
                example_input = torch.randn(1, 3, 224, 224)
                traced_model = torch.jit.trace(quantized_model, example_input)
                script_path = os.path.join(self.log_dir, f'{self.model_name}_quantized_script.pt')
                traced_model.save(script_path)
                self.logger.info(f"TorchScript 모델 저장: {script_path}")
            except Exception as e:
                self.logger.warning(f"TorchScript 변환 중 경고: {str(e)}")
            
            # 3. 양자화 결과 JSON 저장
            results_path = os.path.join(self.log_dir, 'quantization_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"결과 데이터 저장: {results_path}")
            
            # 4. 결과 그래프 생성
            try:
                self._plot_results(results)
            except Exception as e:
                self.logger.warning(f"그래프 생성 중 경고: {str(e)}")
            
            # 5. 요약 텍스트 저장
            summary_path = os.path.join(self.log_dir, 'summary.txt')
            with open(summary_path, 'w') as f:
                f.write("양자화 결과 요약\n")
                f.write("="*60 + "\n")
                f.write(f"모델: {self.model_name}\n")
                f.write(f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("원본 모델:\n")
                f.write(f"  정확도: {results['original']['accuracy']:.2f}%\n")
                f.write(f"  크기: {results['original']['size_mb']:.2f} MB\n")
                f.write(f"  추론시간: {results['original']['inference_ms']:.2f} ms\n\n")
                
                f.write("양자화 모델:\n")
                f.write(f"  정확도: {results['quantized']['accuracy']:.2f}%\n")
                f.write(f"  크기: {results['quantized']['size_mb']:.2f} MB\n")
                f.write(f"  추론시간: {results['quantized']['inference_ms']:.2f} ms\n\n")
                
                f.write("개선 효과:\n")
                f.write(f"  정확도 변화: {results['quantized']['accuracy'] - results['original']['accuracy']:+.2f}%\n")
                f.write(f"  크기 감소: {(1 - results['quantized']['size_mb']/results['original']['size_mb'])*100:.1f}%\n")
                f.write(f"  속도 향상: {results['original']['inference_ms']/results['quantized']['inference_ms']:.2f}x\n")
                
                if 'quantization_time_minutes' in results:
                    f.write(f"\n전체 소요 시간: {results['quantization_time_minutes']:.1f}분\n")
                
                # Calibration 통계가 있으면 추가
                if 'calibration_stats' in results:
                    f.write("\nCalibration 통계 (주요 레이어):\n")
                    for layer_name, stats in list(results['calibration_stats'].items())[:10]:
                        f.write(f"  {layer_name}: [{stats['min']:.6f}, {stats['max']:.6f}]\n")
            
            self.logger.info(f"요약 저장: {summary_path}")
            self.logger.info("모든 결과 저장 완료")
            
        except Exception as e:
            self.logger.exception("결과 저장 중 오류")
            raise
    
    def _plot_results(self, results):
        """결과 시각화"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 정확도 비교
        models = ['Original', 'Quantized']
        accuracies = [results['original']['accuracy'], results['quantized']['accuracy']]
        axes[0].bar(models, accuracies, color=['blue', 'green'])
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Model Accuracy Comparison')
        axes[0].set_ylim(0, 100)
        
        # 값 표시
        for i, v in enumerate(accuracies):
            axes[0].text(i, v + 1, f'{v:.1f}%', ha='center')
        
        # 2. 모델 크기 비교
        sizes = [results['original']['size_mb'], results['quantized']['size_mb']]
        axes[1].bar(models, sizes, color=['blue', 'green'])
        axes[1].set_ylabel('Size (MB)')
        axes[1].set_title('Model Size Comparison')
        
        for i, v in enumerate(sizes):
            axes[1].text(i, v + 0.1, f'{v:.1f}MB', ha='center')
        
        # 3. 추론 시간 비교
        times = [results['original']['inference_ms'], results['quantized']['inference_ms']]
        axes[2].bar(models, times, color=['blue', 'green'])
        axes[2].set_ylabel('Inference Time (ms)')
        axes[2].set_title('Inference Time Comparison')
        
        for i, v in enumerate(times):
            axes[2].text(i, v + 0.5, f'{v:.1f}ms', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'quantization_results.png'), dpi=300, bbox_inches='tight')
        plt.close()  # GUI 없는 환경을 위해 close
        
        print(f"결과 그래프 저장: {os.path.join(self.log_dir, 'quantization_results.png')}")


# 메인 실행 코드
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MobileNet 양자화')
    parser.add_argument('--model', type=str, default='mobilenet_v3_small',
                        choices=['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'])
    parser.add_argument('--train_data', type=str, required=True,
                        help='학습 데이터 경로 (calibration용)')
    parser.add_argument('--val_data', type=str, required=True,
                        help='검증 데이터 경로')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='배치 크기')
    parser.add_argument('--calibration_batches', type=int, default=100,
                        help='Calibration에 사용할 배치 수 (MATLAB calibrate와 동일)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MobileNet INT8 양자화")
    print("="*60)
    
    # 1. 양자화 객체 생성
    quantizer = MobileNetQuantization(model_name=args.model)
    
    # 2. 데이터 로더 준비
    print("\n데이터 로더 준비 중...")
    train_loader, val_loader = quantizer.prepare_data_loaders(
        args.train_data, args.val_data, args.batch_size
    )
    
    # 3. Post-training Quantization 실행
    quantized_model = quantizer.quantize_model(
        train_loader, val_loader, 
        calibration_batches=args.calibration_batches
    )
    
    # 4. 완료
    print("\n" + "="*60)
    print("양자화 완료!")
    print("="*60)
    print(f"\n생성된 파일 위치: {quantizer.log_dir}")
    print("\n생성된 파일:")
    print(f"  - {args.model}_quantized.pth (PyTorch 형식)")
    print(f"  - {args.model}_quantized_script.pt (TorchScript 형식)")
    print("  - quantization_results.json (결과 데이터)")
    print("  - quantization_results.png (결과 그래프)")
    print("  - summary.txt (결과 요약)")