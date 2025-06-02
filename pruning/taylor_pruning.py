# ==================== TaylorPruning 클래스 ====================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from utils import Logger

class TaylorPruning:
    def __init__(self, model, model_name, prunable_layers, device='cuda', log_dir='pruning_logs'):
        self.model = model.to(device)
        self.model_name = model_name
        self.prunable_layers = prunable_layers
        self.device = device
        self.log_dir = log_dir
        
        # 로거 설정
        log_file = os.path.join(log_dir, f"{model_name}_pruning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self.logger = Logger(log_file)
        self.logger.info(f"Taylor 프루닝 시작: {model_name}")
        self.logger.info(f"디바이스: {device}")
        self.logger.info(f"프루닝 가능한 레이어 수: {len(prunable_layers)}")
        
        # 결과 저장용 JSON 파일
        self.results_file = os.path.join(log_dir, f"{model_name}_pruning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.pruning_results = {
            'model_name': model_name,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'iterations': [],
            'final_results': {}
        }
        
        # Taylor 점수와 프루닝 마스크
        self.taylor_scores = {}
        self.pruning_masks = {}
        self.pruning_activations = {}
        self.gradient_hooks = []
        
        # 학습 기록
        self.loss_history = []
        self.accuracy_history = []
        self.num_prunables_history = []
        
        self._initialize_pruning()
        
    def _initialize_pruning(self):
        """프루닝 초기화"""
        self.logger.info("\n프루닝 초기화 중...")
        
        for layer_info in self.prunable_layers:
            layer_name = layer_info['name']
            out_channels = layer_info['out_channels']
            
            # 점수와 마스크 초기화
            self.taylor_scores[layer_name] = torch.zeros(out_channels).to(self.device)
            self.pruning_masks[layer_name] = torch.ones(out_channels, dtype=torch.bool).to(self.device)
            
            self.logger.info(f"  {layer_name}: {out_channels} 필터 초기화")
            
        self.logger.info(f"총 {len(self.taylor_scores)}개 레이어 초기화 완료")
            
    def register_pruning_hooks(self):
        """프루닝을 위한 forward/backward hooks 등록"""
        
        # 기존 hooks 제거
        self.remove_hooks()
        
        # Forward hook: activation 저장
        def create_forward_hook(layer_name):
            def hook_fn(module, input, output):
                # activation 저장 (gradient 계산을 위해)
                self.pruning_activations[layer_name] = output
            return hook_fn
            
        # Backward hook: gradient와 activation을 이용한 Taylor 점수 계산
        def create_backward_hook(layer_name):
            def hook_fn(module, grad_input, grad_output):
                if grad_output[0] is not None and layer_name in self.pruning_activations:
                    # Taylor importance = |gradient * activation|
                    activation = self.pruning_activations[layer_name]
                    gradient = grad_output[0]
                    
                    # 채널별 중요도 계산
                    importance = (gradient * activation).abs().mean(dim=[0, 2, 3])
                    
                    # 현재 활성화된 필터에 대해서만 점수 업데이트
                    mask = self.pruning_masks[layer_name]
                    self.taylor_scores[layer_name][mask] += importance[mask].detach()
                    
            return hook_fn
        
        # Hooks 등록
        for name, module in self.model.named_modules():
            if name in [layer['name'] for layer in self.prunable_layers]:
                # Forward hook
                f_handle = module.register_forward_hook(create_forward_hook(name))
                self.gradient_hooks.append(f_handle)
                
                # Backward hook
                b_handle = module.register_backward_hook(create_backward_hook(name))
                self.gradient_hooks.append(b_handle)
                
    def remove_hooks(self):
        """모든 hooks 제거"""
        for handle in self.gradient_hooks:
            handle.remove()
        self.gradient_hooks = []
        self.pruning_activations = {}
        
    def update_scores(self, train_loader, num_batches=50):
        """Taylor 점수 업데이트"""
        
        # 점수 초기화
        for name in self.taylor_scores:
            self.taylor_scores[name].zero_()
            
        # Hook 등록
        self.register_pruning_hooks()
        
        # 학습 모드
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        
        # 지정된 배치 수만큼 점수 누적
        batch_count = 0
        for inputs, targets in train_loader:
            if batch_count >= num_batches:
                break
                
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass (Taylor 점수 계산)
            self.model.zero_grad()
            loss.backward()
            
            batch_count += 1
            
        # Hook 제거
        self.remove_hooks()
        
        # 점수 정규화 (배치 수로 나눔)
        for name in self.taylor_scores:
            self.taylor_scores[name] /= batch_count
            
    def update_prunables(self, max_to_prune=32):
        """프루닝 마스크 업데이트"""
        
        # 모든 활성 필터의 점수 수집
        all_scores = []
        score_info = []
        
        for layer_name, scores in self.taylor_scores.items():
            mask = self.pruning_masks[layer_name]
            for idx in range(len(scores)):
                if mask[idx]:  # 활성 필터만
                    all_scores.append(scores[idx].item())
                    score_info.append((layer_name, idx))
                    
        # 점수가 낮은 필터 선택
        if len(all_scores) > max_to_prune:
            # 하위 max_to_prune개 선택
            sorted_indices = np.argsort(all_scores)[:max_to_prune]
            
            # 마스크 업데이트
            pruned_count = 0
            pruned_info = {}
            
            for idx in sorted_indices:
                layer_name, filter_idx = score_info[idx]
                self.pruning_masks[layer_name][filter_idx] = False
                pruned_count += 1
                
                # 프루닝 정보 기록
                if layer_name not in pruned_info:
                    pruned_info[layer_name] = []
                pruned_info[layer_name].append(filter_idx)
                
            # 로그에 기록
            self.logger.info(f"프루닝 수행: {pruned_count}개 필터 제거")
            for layer_name, indices in pruned_info.items():
                self.logger.info(f"  {layer_name}: {len(indices)}개 필터 제거")
                
            return pruned_count
        
        return 0
        
    def apply_pruning_masks(self):
        """프루닝 마스크를 실제 가중치에 적용"""
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if name in self.pruning_masks and isinstance(module, nn.Conv2d):
                    mask = self.pruning_masks[name]
                    # 출력 채널에 마스크 적용
                    module.weight.data[~mask] = 0
                    if module.bias is not None:
                        module.bias.data[~mask] = 0
                        
    def fine_tune_step(self, train_loader, optimizer, num_updates=50):
        """Fine-tuning 단계"""
        
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        
        update_count = 0
        total_loss = 0
        
        for inputs, targets in train_loader:
            if update_count >= num_updates:
                break
                
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient에 마스크 적용
            with torch.no_grad():
                for name, module in self.model.named_modules():
                    if name in self.pruning_masks and isinstance(module, nn.Conv2d):
                        mask = self.pruning_masks[name]
                        if module.weight.grad is not None:
                            module.weight.grad[~mask] = 0
                        if module.bias is not None and module.bias.grad is not None:
                            module.bias.grad[~mask] = 0
            
            optimizer.step()
            
            # 프루닝 마스크 재적용
            self.apply_pruning_masks()
            
            total_loss += loss.item()
            update_count += 1
            self.loss_history.append(loss.item())
            
        return total_loss / update_count
        
    def evaluate(self, val_loader):
        """모델 평가"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        accuracy = 100. * correct / total
        return accuracy
        
    def get_num_active_filters(self):
        """활성 필터 수 계산"""
        total = 0
        for mask in self.pruning_masks.values():
            total += mask.sum().item()
        return total
        
    def iterative_pruning(self, train_loader, val_loader, 
                         max_pruning_iterations=30,
                         max_to_prune=32,
                         num_minibatch_updates=50,
                         learning_rate=1e-2/3,
                         momentum=0.9):
        """반복적 프루닝"""
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Taylor 프루닝 시작")
        self.logger.info(f"최대 반복: {max_pruning_iterations}")
        self.logger.info(f"반복당 제거 필터: {max_to_prune}")
        self.logger.info(f"학습률: {learning_rate}")
        self.logger.info(f"모멘텀: {momentum}")
        self.logger.info(f"{'='*60}\n")
        
        # 초기 정확도
        initial_accuracy = self.evaluate(val_loader)
        self.accuracy_history.append(initial_accuracy)
        self.num_prunables_history.append(self.get_num_active_filters())
        self.logger.info(f"초기 정확도: {initial_accuracy:.2f}%")
        self.logger.info(f"초기 활성 필터 수: {self.get_num_active_filters()}")
        
        # Optimizer 설정
        optimizer = optim.SGD(self.model.parameters(), 
                            lr=learning_rate, 
                            momentum=momentum)
        
        # 프루닝 반복
        for pruning_iter in range(max_pruning_iterations):
            self.logger.info(f"\n[Pruning Iteration {pruning_iter+1}/{max_pruning_iterations}]")
            
            iteration_start = datetime.now()
            
            # 1. Fine-tune하면서 Taylor 점수 업데이트
            self.register_pruning_hooks()
            avg_loss = self.fine_tune_step(train_loader, optimizer, num_minibatch_updates)
            self.remove_hooks()
            self.logger.info(f"  평균 손실: {avg_loss:.4f}")
            
            # 2. Taylor 점수 계산
            self.update_scores(train_loader, num_batches=10)
            
            # 3. 프루닝 수행
            pruned = self.update_prunables(max_to_prune)
            self.apply_pruning_masks()
            
            # 4. 평가
            accuracy = self.evaluate(val_loader)
            num_active = self.get_num_active_filters()
            
            self.accuracy_history.append(accuracy)
            self.num_prunables_history.append(num_active)
            
            # 반복 결과 기록
            iteration_time = (datetime.now() - iteration_start).total_seconds()
            iteration_result = {
                'iteration': pruning_iter + 1,
                'pruned_filters': pruned,
                'active_filters': num_active,
                'accuracy': accuracy,
                'avg_loss': avg_loss,
                'time_seconds': iteration_time
            }
            self.pruning_results['iterations'].append(iteration_result)
            
            self.logger.info(f"  제거된 필터: {pruned}")
            self.logger.info(f"  활성 필터: {num_active}")
            self.logger.info(f"  정확도: {accuracy:.2f}%")
            self.logger.info(f"  소요 시간: {iteration_time:.2f}초")
            
            # 조기 종료 조건
            if pruned == 0:
                self.logger.info("\n더 이상 제거할 필터가 없습니다.")
                break
                
        # 최종 결과 기록
        final_accuracy = self.accuracy_history[-1]
        self.pruning_results['final_results'] = {
            'initial_accuracy': initial_accuracy,
            'final_accuracy': final_accuracy,
            'accuracy_change': final_accuracy - initial_accuracy,
            'initial_filters': self.num_prunables_history[0],
            'final_filters': self.num_prunables_history[-1],
            'compression_ratio': 1 - (self.num_prunables_history[-1] / self.num_prunables_history[0]),
            'total_iterations': len(self.pruning_results['iterations'])
        }
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"프루닝 완료!")
        self.logger.info(f"초기 정확도: {initial_accuracy:.2f}%")
        self.logger.info(f"최종 정확도: {final_accuracy:.2f}% (변화: {final_accuracy - initial_accuracy:+.2f}%)")
        self.logger.info(f"필터 압축률: {self.pruning_results['final_results']['compression_ratio']*100:.1f}%")
        self.logger.info(f"{'='*60}")
        
        # 결과 JSON 파일로 저장
        with open(self.results_file, 'w') as f:
            json.dump(self.pruning_results, f, indent=2)
        self.logger.info(f"\n결과 저장: {self.results_file}")
        
    def plot_pruning_progress(self):
        """프루닝 진행 상황 시각화"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Loss plot
        ax1.plot(self.loss_history)
        ax1.set_xlabel('Fine-tuning Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Mini-Batch Loss During Pruning')
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.accuracy_history, 'o-')
        ax2.set_xlabel('Pruning Iteration')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy After Pruning')
        ax2.grid(True)
        
        # Number of filters plot
        ax3.plot(self.num_prunables_history, '^-')
        ax3.set_xlabel('Pruning Iteration')
        ax3.set_ylabel('Active Filters')
        ax3.set_title('Number of Active Convolution Filters')
        ax3.grid(True)
        
        plt.tight_layout()
        
        # 그래프 저장
        plot_file = os.path.join(self.log_dir, f"{self.model_name}_pruning_progress.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        self.logger.info(f"그래프 저장: {plot_file}")
        
        plt.show()
        
    def save_pruned_model(self, save_path=None):
        """프루닝된 모델 저장"""
        if save_path is None:
            save_path = os.path.join(self.log_dir, f'{self.model_name}_pruned.pth')
            
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'pruning_masks': self.pruning_masks,
            'taylor_scores': self.taylor_scores,
            'accuracy_history': self.accuracy_history,
            'loss_history': self.loss_history,
            'num_prunables_history': self.num_prunables_history,
            'model_name': self.model_name,
            'final_results': self.pruning_results['final_results']
        }
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"프루닝된 모델 저장: {save_path}")
        
        # 레이어별 프루닝 통계 저장
        layer_stats_file = os.path.join(self.log_dir, f"{self.model_name}_layer_statistics.txt")
        with open(layer_stats_file, 'w') as f:
            f.write(f"레이어별 프루닝 통계\n")
            f.write(f"{'='*60}\n")
            f.write(f"{'Layer Name':<30} {'Total':<10} {'Active':<10} {'Pruned':<10} {'Ratio(%)':<10}\n")
            f.write(f"{'-'*60}\n")
            
            for layer_name, mask in self.pruning_masks.items():
                total = len(mask)
                active = mask.sum().item()
                pruned = total - active
                ratio = (pruned / total) * 100
                f.write(f"{layer_name:<30} {total:<10} {active:<10} {pruned:<10} {ratio:<10.1f}\n")
                
        self.logger.info(f"레이어별 통계 저장: {layer_stats_file}")
        
    def __del__(self):
        if hasattr(self, 'logger'):
            self.logger.close()
