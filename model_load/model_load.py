"""
Step 1: 모델 로드 및 분석
MATLAB의 analyzeNetwork 기능
에러 처리 및 로깅 기능 포함
"""

import torch
import torch.nn as nn
import torchvision.models as models
import os
from datetime import datetime
import traceback
import sys

class Logger:
    """콘솔과 파일에 동시에 출력하는 로거"""
    def __init__(self, log_file):
        self.terminal = print
        self.log_file = log_file
        
        # 로그 디렉토리 생성
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 로그 파일 열기 (append 모드)
        self.file = open(log_file, 'a', encoding='utf-8')
        
        # 시작 시간 기록
        self.file.write(f"\n\n{'='*80}\n")
        self.file.write(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file.write(f"Python 버전: {sys.version}\n")
        self.file.write(f"PyTorch 버전: {torch.__version__}\n")
        self.file.write(f"CUDA 사용 가능: {torch.cuda.is_available()}\n")
        self.file.write(f"{'='*80}\n")
        
    def write(self, message, level='INFO'):
        """콘솔과 파일에 동시 출력"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] [{level}] {message}"
        
        # 콘솔 출력 (레벨에 따라 색상 변경 가능)
        if level == 'ERROR':
            self.terminal(f"\033[91m{formatted_message}\033[0m")  # 빨간색
        elif level == 'WARNING':
            self.terminal(f"\033[93m{formatted_message}\033[0m")  # 노란색
        else:
            self.terminal(formatted_message)
        
        # 파일 출력
        self.file.write(formatted_message + '\n')
        self.file.flush()
        
    def error(self, message):
        """에러 메시지 출력"""
        self.write(message, 'ERROR')
        
    def warning(self, message):
        """경고 메시지 출력"""
        self.write(message, 'WARNING')
        
    def info(self, message):
        """정보 메시지 출력"""
        self.write(message, 'INFO')
        
    def exception(self, message, exc_info=None):
        """예외 정보를 포함한 에러 로깅"""
        self.error(message)
        if exc_info:
            # 예외 상세 정보 기록
            tb_str = ''.join(traceback.format_exception(*exc_info))
            self.file.write(f"\n예외 상세 정보:\n{tb_str}\n")
            self.file.flush()
            # 콘솔에도 출력
            self.terminal(f"\033[91m{tb_str}\033[0m")
        
    def close(self):
        """파일 닫기"""
        self.file.close()

def load_and_analyze_model(model_name='mobilenet_v2', log_dir='/home/minha/raspberrypi/model_load'):
    """모델 로드 및 분석 (에러 처리 포함)"""
    
    # 로그 파일 설정
    log_file = os.path.join(log_dir, f"{model_name}_analysis.log")
    error_log_file = os.path.join(log_dir, f"{model_name}_errors.log")
    logger = Logger(log_file)
    error_logger = Logger(error_log_file)
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"{model_name} 로드 및 분석 시작")
        logger.info(f"{'='*60}")
        
        # 1. 모델 로드
        logger.info(f"\n1. 모델 로드 중...")
        try:
            if model_name == 'mobilenet_v3_small':
                model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
            elif model_name == 'mobilenet_v3_large':
                model = models.mobilenet_v3_large(weights='IMAGENET1K_V2')
            elif model_name == 'mobilenet_v2':
                model = models.mobilenet_v2(weights='IMAGENET1K_V2')
            else:
                raise ValueError(f"지원하지 않는 모델: {model_name}")
            
            model.eval()
            logger.info(f"✓ {model_name} 로드 완료")
            
        except Exception as e:
            error_msg = f"모델 로드 실패: {model_name}"
            logger.error(error_msg)
            error_logger.exception(error_msg, sys.exc_info())
            raise
        
        # 2. 전체 파라미터 수 계산
        logger.info(f"\n2. 전체 파라미터 분석")
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info(f"총 파라미터 수: {total_params:,}")
            logger.info(f"학습 가능한 파라미터: {trainable_params:,}")
            logger.info(f"모델 크기 (MB): {total_params * 4 / (1024*1024):.2f}")
            
            if total_params == 0:
                raise ValueError("모델에 파라미터가 없습니다")
                
        except Exception as e:
            error_msg = "파라미터 분석 중 오류 발생"
            logger.error(error_msg)
            error_logger.exception(error_msg, sys.exc_info())
            raise
        
        # 3. Conv2d 레이어 분석
        logger.info(f"\n3. Conv2d 레이어 상세 분석")
        conv_layers_info = []
        
        try:
            logger.info(f"{'Layer Name':<40} {'In':<6} {'Out':<6} {'Kernel':<10} {'Groups':<8} {'Params':<10}")
            logger.info("-" * 90)
            
            conv_count = 0
            total_conv_params = 0
            
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    try:
                        conv_count += 1
                        params = module.weight.numel()
                        if module.bias is not None:
                            params += module.bias.numel()
                        total_conv_params += params
                        
                        layer_info = {
                            'name': name,
                            'in_channels': module.in_channels,
                            'out_channels': module.out_channels,
                            'kernel_size': module.kernel_size,
                            'groups': module.groups,
                            'params': params
                        }
                        conv_layers_info.append(layer_info)
                        
                        logger.info(f"{name:<40} {module.in_channels:<6} {module.out_channels:<6} "
                                   f"{str(module.kernel_size):<10} {module.groups:<8} {params:<10,}")
                        
                    except Exception as e:
                        warning_msg = f"레이어 분석 중 경고: {name} - {str(e)}"
                        logger.warning(warning_msg)
                        error_logger.warning(warning_msg)
                        continue
            
            logger.info(f"\n총 Conv2d 레이어 수: {conv_count}")
            logger.info(f"Conv2d 레이어 총 파라미터: {total_conv_params:,} ({total_conv_params/total_params*100:.1f}%)")
            
        except Exception as e:
            error_msg = "Conv2d 레이어 분석 중 오류 발생"
            logger.error(error_msg)
            error_logger.exception(error_msg, sys.exc_info())
            raise
        
        # 4. 프루닝 가능한 레이어 찾기
        logger.info(f"\n4. 프루닝 가능한 레이어 분석")
        prunable_layers = []
        
        try:
            for layer in conv_layers_info:
                try:
                    module = dict(model.named_modules())[layer['name']]
                    # Depthwise conv (groups == in_channels)는 보통 프루닝 제외
                    if module.groups == 1 or module.groups != module.in_channels:
                        prunable_layers.append(layer)
                except KeyError as e:
                    warning_msg = f"레이어 접근 오류: {layer['name']}"
                    logger.warning(warning_msg)
                    error_logger.warning(f"{warning_msg} - {str(e)}")
                    continue
            
            if len(prunable_layers) == 0:
                logger.warning("프루닝 가능한 레이어가 없습니다!")
            else:
                logger.info(f"프루닝 가능한 레이어 수: {len(prunable_layers)}")
                logger.info(f"프루닝 가능한 파라미터: {sum(l['params'] for l in prunable_layers):,}")
                logger.info("\n프루닝 가능한 레이어 목록:")
                
                for i, layer in enumerate(prunable_layers):
                    logger.info(f"  [{i+1}] {layer['name']} - "
                               f"out_channels: {layer['out_channels']}, "
                               f"params: {layer['params']:,}")
                    
        except Exception as e:
            error_msg = "프루닝 가능 레이어 분석 중 오류 발생"
            logger.error(error_msg)
            error_logger.exception(error_msg, sys.exc_info())
            raise
        
        # 5. 레이어 타입별 통계
        logger.info(f"\n5. 레이어 타입별 통계")
        try:
            layer_types = {}
            for name, module in model.named_modules():
                layer_type = type(module).__name__
                if layer_type not in layer_types:
                    layer_types[layer_type] = 0
                layer_types[layer_type] += 1
            
            for layer_type, count in sorted(layer_types.items(), key=lambda x: x[1], reverse=True):
                if count > 1:
                    logger.info(f"  {layer_type}: {count}")
                    
        except Exception as e:
            warning_msg = "레이어 타입 통계 중 경고 발생"
            logger.warning(warning_msg)
            error_logger.warning(f"{warning_msg}: {str(e)}")
        
        # 6. 분석 요약
        logger.info(f"\n6. 분석 요약")
        logger.info(f"모델: {model_name}")
        logger.info(f"총 레이어 수: {len(list(model.modules()))}")
        logger.info(f"총 파라미터: {total_params:,}")
        logger.info(f"모델 크기: {total_params * 4 / (1024*1024):.2f} MB")
        logger.info(f"Conv2d 레이어: {conv_count}개")
        logger.info(f"프루닝 가능 레이어: {len(prunable_layers)}개")
        
        logger.info(f"\n✓ 분석 완료!")
        logger.info(f"로그 파일: {log_file}")
        logger.info(f"에러 로그: {error_log_file}")
        
        return model, prunable_layers, conv_layers_info
        
    except Exception as e:
        error_msg = f"\n심각한 오류 발생! 분석 중단됨"
        logger.error(error_msg)
        error_logger.exception(error_msg, sys.exc_info())
        
        # 정리
        logger.error(f"분석 실패. 에러 로그 확인: {error_log_file}")
        return None, None, None
        
    finally:
        # 로거 닫기
        logger.close()
        error_logger.close()

# 실행
if __name__ == "__main__":
    # 어떤 모델 분석할지 선택
    model_name = 'mobilenet_v2'  # 이거 바꿔가면서 실행
    
    print(f"\n{model_name} 분석 시작...")
    
    try:
        model, prunable_layers, conv_info = load_and_analyze_model(model_name)
        
        if model is not None:
            print(f"\n✓ 분석 성공!")
            print(f"다음 단계: Taylor 프루닝 준비")
            print(f"프루닝 가능한 레이어 수: {len(prunable_layers)}")
        else:
            print(f"\n✗ 분석 실패! 에러 로그를 확인하세요.")
            
    except Exception as e:
        print(f"\n✗ 예상치 못한 오류 발생: {str(e)}")
        print(f"상세 내용은 에러 로그를 확인하세요.")