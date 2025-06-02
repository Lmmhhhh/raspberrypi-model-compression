"""
utils.py - 로거 및 유틸리티 함수
"""
import os
import sys
from datetime import datetime

class Logger:
    """로그를 파일과 콘솔에 동시 출력"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.file = open(log_file, 'a', encoding='utf-8')
        
        # 헤더 작성
        self.file.write(f"\n{'='*80}\n")
        self.file.write(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file.write(f"{'='*80}\n")
        
    def write(self, message, level='INFO'):
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] [{level}] {message}"
        
        # 콘솔 출력
        if level == 'ERROR':
            print(f"\033[91m{formatted_message}\033[0m")
        elif level == 'WARNING':
            print(f"\033[93m{formatted_message}\033[0m")
        else:
            print(formatted_message)
            
        # 파일 출력
        self.file.write(formatted_message + '\n')
        self.file.flush()
        
    def info(self, message): self.write(message, 'INFO')
    def warning(self, message): self.write(message, 'WARNING')
    def error(self, message): self.write(message, 'ERROR')
    
    def close(self):
        self.file.close()

def create_experiment_dir(base_dir, model_name):
    """실험별 디렉토리 생성"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"pruning_{model_name}_{timestamp}"
    
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 하위 디렉토리 생성
    subdirs = {
        'logs': os.path.join(experiment_dir, 'logs'),
        'models': os.path.join(experiment_dir, 'models'),
        'plots': os.path.join(experiment_dir, 'plots'),
        'results': os.path.join(experiment_dir, 'results')
    }
    
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    return experiment_dir, subdirs