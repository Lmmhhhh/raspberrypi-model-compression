"""
main_pruning.py - MobileNet 프루닝 메인 실행 파일
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from datetime import datetime
import json
import os

from taylor_pruning import TaylorPruning
from utils import create_experiment_dir

# 기본 경로 설정
DEFAULT_PATHS = {
    'train_data': '/home/minha/.cache/kagglehub/datasets/tusonggao/imagenet-train-subset-100k/versions/1/imagenet_subtrain',
    'val_data': '/home/minha/.cache/kagglehub/datasets/tusonggao/imagenet-validation-dataset/versions/1/imagenet_validation',
    'output_base': '/home/minha/raspberrypi/pruning/mobilenet_v3_small',
}

# MobileNet 모델 설정
MOBILENET_CONFIGS = {
    'mobilenet_v3_small': {
        'load_fn': models.mobilenet_v3_small,
        'weights': 'IMAGENET1K_V1',
        'batch_size': 32,
        'learning_rate': 1e-2/3,
    }
}

"""
    mobilenet_v2': {
        'load_fn': models.mobilenet_v2,
        'weights': 'IMAGENET1K_V2',
        'batch_size': 32,
        'learning_rate': 1e-3,
    },
    'mobilenet_v3_small': {
        'load_fn': models.mobilenet_v3_small,
        'weights': 'IMAGENET1K_V1',
        'batch_size': 64,
        'learning_rate': 1e-2/3,
    },
    'mobilenet_v3_large': {
        'load_fn': models.mobilenet_v3_large,
        'weights': 'IMAGENET1K_V2',
        'batch_size': 32,  # 더 큰 모델
        'learning_rate': 1e-3,
    }
"""

def load_mobilenet(model_name):
    """MobileNet 모델 로드"""
    if model_name not in MOBILENET_CONFIGS:
        raise ValueError(f"지원하지 않는 모델: {model_name}. 지원 모델: {list(MOBILENET_CONFIGS.keys())}")
    
    config = MOBILENET_CONFIGS[model_name]
    print(f"모델 로드 중: {model_name}")
    
    # 모델 로드
    model = config['load_fn'](weights=config['weights'])
    return model, config

def get_prunable_layers(model):
    """모델에서 프루닝 가능한 레이어 추출"""
    prunable_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Depthwise convolution 제외 (groups == in_channels)
            if module.groups == 1 or module.groups != module.in_channels:
                layer_info = {
                    'name': name,
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size,
                    'groups': module.groups
                }
                prunable_layers.append(layer_info)
    
    return prunable_layers

def prepare_data_loaders(train_path, val_path, batch_size=32, num_workers=4):
    """ImageNet 데이터 로더 준비"""
    
    print(f"학습 데이터 경로: {train_path}")
    print(f"검증 데이터 경로: {val_path}")
    
    # ImageNet 전처리
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 로드
    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    val_dataset = datasets.ImageFolder(val_path, transform=transform)
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=num_workers)
    
    print(f"학습 데이터: {len(train_dataset)} 샘플")
    print(f"검증 데이터: {len(val_dataset)} 샘플")
    
    return train_loader, val_loader

def main(args):
    """메인 실행 함수"""
    
    # 실험 디렉토리 생성
    experiment_dir, subdirs = create_experiment_dir(args.output_dir, args.model)
    print(f"실험 결과 저장 경로: {experiment_dir}")
    
    # 1. 모델 로드
    print("\n" + "="*60)
    print(f"Step 1: 모델 로드 및 분석 - {args.model}")
    print("="*60)
    
    model, config = load_mobilenet(args.model)
    model.eval()
    
    # 프루닝 가능한 레이어 추출
    prunable_layers = get_prunable_layers(model)
    print(f"프루닝 가능한 레이어 수: {len(prunable_layers)}")
    
    # 총 필터 수 계산
    total_filters = sum(layer['out_channels'] for layer in prunable_layers)
    print(f"총 프루닝 가능한 필터 수: {total_filters}")
    
    # 2. 데이터 로더 준비
    print("\n" + "="*60)
    print("Step 2: 데이터 준비")
    print("="*60)
    
    train_loader, val_loader = prepare_data_loaders(
        args.train_data, 
        args.val_data,
        batch_size=config['batch_size'],
        num_workers=args.num_workers
    )
    
    # 3. 프루닝 설정
    print("\n" + "="*60)
    print("Step 3: 프루닝 설정")
    print("="*60)
    
    target_compression = args.compression
    max_to_prune = int(total_filters * target_compression / args.iterations)
    
    print(f"목표 압축률: {target_compression*100}%")
    print(f"프루닝 반복 횟수: {args.iterations}")
    print(f"반복당 제거할 필터 수: {max_to_prune}")
    print(f"예상 총 제거 필터: {max_to_prune * args.iterations}")
    print(f"학습률: {config['learning_rate']}")
    print(f"배치 크기: {config['batch_size']}")
    
    # 4. Taylor 프루닝 실행
    print("\n" + "="*60)
    print("Step 4: Taylor 프루닝 실행")
    print("="*60)
    
    pruner = TaylorPruning(
        model=model,
        model_name=args.model,
        prunable_layers=prunable_layers,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_dir=subdirs['logs']
    )
    
    # 프루닝 실행
    pruner.iterative_pruning(
        train_loader=train_loader,
        val_loader=val_loader,
        max_pruning_iterations=args.iterations,
        max_to_prune=max_to_prune,
        num_minibatch_updates=50,
        learning_rate=config['learning_rate'],
        momentum=0.9
    )
    
    # 5. 결과 저장
    print("\n" + "="*60)
    print("Step 5: 결과 저장")
    print("="*60)
    
    # 그래프 저장
    pruner.log_dir = subdirs['plots']
    pruner.plot_pruning_progress()
    
    # 모델 저장
    model_save_path = os.path.join(subdirs['models'], f'{args.model}_pruned.pth')
    pruner.save_pruned_model(model_save_path)
    
    # 실험 요약 저장
    summary_file = os.path.join(experiment_dir, 'experiment_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"실험 요약\n")
        f.write(f"="*60 + "\n")
        f.write(f"실험 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"모델: {args.model}\n")
        f.write(f"목표 압축률: {target_compression*100}%\n")
        f.write(f"프루닝 반복: {args.iterations}회\n")
        f.write(f"학습률: {config['learning_rate']}\n")
        f.write(f"배치 크기: {config['batch_size']}\n")
        
        if hasattr(pruner, 'pruning_results'):
            final_results = pruner.pruning_results['final_results']
            f.write(f"\n최종 결과:\n")
            f.write(f"초기 정확도: {final_results['initial_accuracy']:.2f}%\n")
            f.write(f"최종 정확도: {final_results['final_accuracy']:.2f}%\n")
            f.write(f"정확도 변화: {final_results['accuracy_change']:+.2f}%\n")
            f.write(f"필터 압축률: {final_results['compression_ratio']*100:.1f}%\n")
    
    print(f"\n모든 결과가 저장되었습니다: {experiment_dir}")
    
    return pruner, experiment_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MobileNet Taylor 프루닝')
    
    parser.add_argument('--model', type=str, default='mobilenet_v3_small',
                        choices=['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'],
                        help='프루닝할 MobileNet 모델')
    parser.add_argument('--train_data', type=str, default=DEFAULT_PATHS['train_data'],
                        help='학습 데이터 경로')
    parser.add_argument('--val_data', type=str, default=DEFAULT_PATHS['val_data'],
                        help='검증 데이터 경로')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_PATHS['output_base'],
                        help='출력 디렉토리')
    parser.add_argument('--compression', type=float, default=0.5,
                        help='목표 압축률 (0-1)')
    parser.add_argument('--iterations', type=int, default=30,
                        help='프루닝 반복 횟수')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='데이터 로더 워커 수')
    
    args = parser.parse_args()
    
    # 메인 함수 실행
    pruner, experiment_dir = main(args)
    
    print("\n" + "="*60)
    print("프루닝 완료!")
    print(f"결과 디렉토리: {experiment_dir}")
    print("="*60)