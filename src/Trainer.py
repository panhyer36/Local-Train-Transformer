import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class TransformerTrainer:
    """
    Transformer模型訓練器類
    """
    
    def _get_best_device(self):
        """自動選擇最佳設備"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def __init__(self, model, dataset, train_ratio=0.8, batch_size=32, 
                 learning_rate=1e-4, device=None, show_plot=True, config=None, save_dir=".", train_indices=None, val_indices=None, test_indices=None):
        """
        初始化訓練器
        
        Args:
            model: TransformerModel實例
            dataset: 數據集
            train_ratio: 訓練集比例
            batch_size: 批次大小
            learning_rate: 學習率
            device: 運算設備
            show_plot: 是否顯示圖表
            config: 配置對象
            save_dir: 圖像和模型儲存目錄
            train_indices: 訓練集索引
            val_indices: 驗證集索引
            test_indices: 測試集索引
        """
        self.model = model
        self.config = config
        self.device = device or self._get_best_device()
        self.model.to(self.device)
        self.show_plot = show_plot
        self.save_dir = save_dir
        
        # 分割數據集（使用順序拆分）
        if train_indices is not None and val_indices is not None:
            # 使用提供的索引
            self.train_dataset = Subset(dataset, train_indices)
            self.val_dataset = Subset(dataset, val_indices)
            if test_indices is not None:
                self.test_dataset = Subset(dataset, test_indices)
                print(f"使用提供的索引: 訓練集 {len(train_indices)} 個序列, 驗證集 {len(val_indices)} 個序列, 測試集 {len(test_indices)} 個序列")
            else:
                self.test_dataset = None
                print(f"使用提供的索引: 訓練集 {len(train_indices)} 個序列, 驗證集 {len(val_indices)} 個序列")
        else:
            # 使用順序拆分（時間序列數據的正確方式）
            total_size = len(dataset)
            train_size = int(train_ratio * total_size)
            
            train_indices = list(range(train_size))
            val_indices = list(range(train_size, total_size))
            
            self.train_dataset = Subset(dataset, train_indices)
            self.val_dataset = Subset(dataset, val_indices)
            self.test_dataset = None
            print(f"順序拆分: 訓練集 {len(train_indices)} 個序列, 驗證集 {len(val_indices)} 個序列")
        
        # 創建數據加載器
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, 
            shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, 
            shuffle=False, num_workers=0
        )
        if self.test_dataset is not None:
            self.test_loader = DataLoader(
                self.test_dataset, batch_size=batch_size, 
                shuffle=False, num_workers=0
            )
        else:
            self.test_loader = None
        
        # 設置優化器和損失函數
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 損失函數（可根據任務調整）
        self.criterion = nn.MSELoss()  # 回歸任務
        
        # 記錄訓練歷史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        print(f"訓練器初始化完成:")
        print(f"設備: {self.device}")
        print(f"訓練樣本數: {len(self.train_dataset)}")
        print(f"驗證樣本數: {len(self.val_dataset)}")
        print(f"模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"是否打印圖表: {self.show_plot}")

        # 設置TensorBoard
        if self.config and hasattr(self.config, 'logging') and self.config.logging.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.config.logging.log_dir)
        else:
            self.writer = None
    
    def train_epoch(self):
        """訓練一個epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for inputs, targets in progress_bar:
            inputs = inputs.float().to(self.device)
            targets = targets.float().to(self.device)
            
            # 前向傳播 - 模型現在直接輸出(batch_size, 1)
            predictions = self.model(inputs)
            
            # 計算損失 - 確保尺寸匹配
            loss = self.criterion(predictions.squeeze(-1), targets)
            
            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新進度條
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / num_batches
    
    def validate(self):
        """驗證模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validating"):
                inputs = inputs.float().to(self.device)
                targets = targets.float().to(self.device)
                
                # 前向傳播 - 模型現在直接輸出(batch_size, 1)
                predictions = self.model(inputs)
                
                # 計算損失 - 確保尺寸匹配
                loss = self.criterion(predictions.squeeze(-1), targets)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, num_epochs, save_path="best_model.pth", early_stopping_patience=10, start_epoch=0):
        """
        訓練模型
        
        Args:
            num_epochs: 訓練輪數
            save_path: 模型保存路徑
            early_stopping_patience: 早停耐心值
            start_epoch: 起始epoch（用於恢復訓練）
        """
        print(f"\n開始訓練 {num_epochs} 個epochs...")
        
        patience_counter = 0
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 訓練
            train_loss = self.train_epoch()
            
            # 驗證
            val_loss = self.validate()
            
            # 記錄損失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 學習率調度
            self.scheduler.step(val_loss)
            
            print(f"訓練損失: {train_loss:.6f}")
            print(f"驗證損失: {val_loss:.6f}")
            print(f"當前學習率: {self.optimizer.param_groups[0]['lr']:.2e}")

            # 在train方法中
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Learning_rate', 
                                    self.optimizer.param_groups[0]['lr'], epoch)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_path)
                print(f"✓ 保存最佳模型 (驗證損失: {val_loss:.6f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 早停檢查
            if patience_counter >= early_stopping_patience:
                print(f"\n早停觸發！{early_stopping_patience} 個epochs內驗證損失未改善")
                break
        
        print("\n訓練完成！")
        return self.train_losses, self.val_losses
    

    def sMAPE(self, y_true, y_pred):
        """計算sMAPE"""
        if y_true == 0 and y_pred == 0:
            return 0
        return 2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)) * 100

    def plot_summary(self):

        self.plot_training_history()
        self.plot_predictions()
        self.plot_perfect_prediction()
        self.plot_attention_weights()
        self.plot_error_percentage_summary()
        self.plot_sMAPE_summary()

    def plot_predictions(self):
        """繪製預測結果與實際值折線圖並儲存"""
        self.model.eval()
        all_targets = []
        all_predictions = []
        
        # 優先使用測試集，如果沒有測試集則使用驗證集
        data_loader = self.test_loader if self.test_loader is not None else self.val_loader
        dataset_name = "Test Set" if self.test_loader is not None else "Validation Set"
        filename = "test_predictions.png" if self.test_loader is not None else "val_predictions.png"
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.float().to(self.device)
                targets = targets.float().to(self.device)
                predictions = self.model(inputs)  # 直接輸出(batch_size, 1)
                all_targets.append(targets.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
        all_targets = np.concatenate(all_targets, axis=0).flatten()
        all_predictions = np.concatenate(all_predictions, axis=0).flatten()

        # 使用所有數據點，不進行採樣
        n_samples = len(all_predictions)
        
        # 根據數據量調整圖形尺寸
        # 數據點越多，圖形越寬，便於觀察細節
        fig_width = max(15, min(30, n_samples // 100))  # 動態調整寬度，最小15，最大30
        plt.figure(figsize=(fig_width, 6))
        
        # 繪製真實值和預測值
        plt.plot(range(n_samples), all_targets, 
                 label='True Values', linewidth=1.0, alpha=0.8)
        plt.plot(range(n_samples), all_predictions, 
                 label='Predictions', linewidth=1.0, alpha=0.8)
        
        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.title(f'Complete Time Series Prediction Comparison - {dataset_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        if self.show_plot:
            plt.show()
        plt.close()

    def plot_perfect_prediction(self):
        """繪製完美預測線與預測值點圖並儲存"""
        self.model.eval()
        all_targets = []
        all_predictions = []
        
        # 優先使用測試集，如果沒有測試集則使用驗證集
        data_loader = self.test_loader if self.test_loader is not None else self.val_loader
        dataset_name = "Test Set" if self.test_loader is not None else "Validation Set"
        filename = "test_perfect_prediction.png" if self.test_loader is not None else "perfect_prediction.png"
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.float().to(self.device)
                targets = targets.float().to(self.device)
                predictions = self.model(inputs)  # 直接輸出(batch_size, 1)
                all_targets.append(targets.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
        all_targets = np.concatenate(all_targets, axis=0).flatten()
        all_predictions = np.concatenate(all_predictions, axis=0).flatten()

        plt.figure(figsize=(6, 6))
        plt.scatter(all_targets, all_predictions, alpha=0.5, color='green', label='Predicted Points')
        min_val = min(all_targets.min(), all_predictions.min())
        max_val = max(all_targets.max(), all_predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction Line')
        plt.title(f'{dataset_name}: Predicted vs. Actual (Perfect Prediction Line)')
        plt.xlabel('Actual Power Demand')
        plt.ylabel('Predicted Power Demand')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        if self.show_plot:
            plt.show()
        plt.close()

    def plot_training_history(self):
        """繪製訓練歷史"""
        if not self.train_losses:
            print("沒有訓練歷史可繪製")
            return
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_losses, color='red', linewidth=2)
        plt.title('Validation Loss Trend')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        if self.show_plot:
            plt.show()
        plt.close()
    
    def plot_attention_weights(self):
        """可視化注意力權重，顯示模型關注的時間步"""
        self.model.eval()
        
        # 取一個批次的數據來分析
        with torch.no_grad():
            for inputs, _ in self.val_loader:
                inputs = inputs.float().to(self.device)
                
                # 獲取注意力權重
                attention_weights = self.model.get_attention_weights(inputs)
                # 轉換維度和格式: (batch_size, seq_len, 1) -> (batch_size, seq_len)
                attention_weights = attention_weights.squeeze(-1).cpu().numpy()
                
                # 只取前5個樣本進行可視化
                num_samples = min(5, attention_weights.shape[0])
                
                plt.figure(figsize=(15, 3 * num_samples))
                for i in range(num_samples):
                    plt.subplot(num_samples, 1, i + 1)
                    weights = attention_weights[i]
                    
                    # 繪製注意力權重
                    plt.plot(range(len(weights)), weights, 'b-', linewidth=2)
                    plt.fill_between(range(len(weights)), weights, alpha=0.3)
                    plt.title(f'Sample {i+1}: Attention Weights Across Time Steps')
                    plt.xlabel('Time Step')
                    plt.ylabel('Attention Weight')
                    plt.grid(True, alpha=0.3)
                    
                    # 標記最高權重的時間步
                    max_idx = np.argmax(weights)
                    plt.scatter(max_idx, weights[max_idx], color='red', s=100, zorder=5)
                    plt.text(max_idx, weights[max_idx], f'Max: {weights[max_idx]:.3f}', 
                            verticalalignment='bottom', horizontalalignment='center')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_dir, 'attention_weights.png'))
                if self.show_plot:
                    plt.show()
                plt.close()
                
                # 計算平均注意力權重
                avg_weights = attention_weights.mean(axis=0)
                plt.figure(figsize=(12, 4))
                plt.plot(range(len(avg_weights)), avg_weights, 'r-', linewidth=3)
                plt.fill_between(range(len(avg_weights)), avg_weights, alpha=0.3, color='red')
                plt.title('Average Attention Weights Across All Validation Samples')
                plt.xlabel('Time Step')
                plt.ylabel('Average Attention Weight')
                plt.grid(True, alpha=0.3)
                
                # 標記最重要的時間步
                top_5_indices = np.argsort(avg_weights)[-5:]
                for idx in top_5_indices:
                    plt.scatter(idx, avg_weights[idx], color='blue', s=60, zorder=5)
                    plt.text(idx, avg_weights[idx], f'{idx}', 
                            verticalalignment='bottom', horizontalalignment='center')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_dir, 'average_attention_weights.png'))
                if self.show_plot:
                    plt.show()
                plt.close()
                
                print(f"注意力權重統計:")
                print(f"最重要的5個時間步: {top_5_indices}")
                print(f"對應的平均權重: {avg_weights[top_5_indices]}")
                
                break  # 只處理第一個批次
    
    def plot_error_percentage_summary(self):
        """
        plot 實際值與預測值的百分比誤差折線圖(1)和直方圖(2)
        """
        self.model.eval()
        all_targets = []
        all_predictions = []
        
        # 優先使用測試集，如果沒有測試集則使用驗證集
        data_loader = self.test_loader if self.test_loader is not None else self.val_loader
        dataset_name = "Test Set" if self.test_loader is not None else "Validation Set"
        filename_prefix = "test" if self.test_loader is not None else "val"
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.float().to(self.device)
                targets = targets.float().to(self.device)
                predictions = self.model(inputs)
                all_targets.append(targets.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())

        all_targets = np.concatenate(all_targets, axis=0).flatten()
        all_predictions = np.concatenate(all_predictions, axis=0).flatten()

        # calculate error percentage
        error_percentage = ((all_predictions - all_targets) / (all_targets + 1e-8)) * 100

        # 1.折線圖
        plt.figure(figsize=(12, 6))
        plt.plot(error_percentage, color='purple', alpha=0.6, label='Error Percentage')
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.title(f'Prediction Error Percentage on {dataset_name}')
        plt.xlabel('Sample Index')
        plt.ylabel('Error (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{filename_prefix}_error_percentage_line.png'))
        if self.show_plot:
            plt.show()
        plt.close()

        # 2.直方圖
        plt.figure(figsize=(8, 6))
        plt.hist(error_percentage, bins=100, color='orange', alpha=0.7, edgecolor='black')
        plt.title(f'Prediction Error Percentage Distribution ({dataset_name})')
        plt.xlabel('Error (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{filename_prefix}_error_percentage_histogram.png'))
        if self.show_plot:
            plt.show()
        plt.close()

        # print error percentage statistics
        print(f"Error Percentage Statistics:")
        print(f" Mean: {error_percentage.mean():.2f}%")
        print(f" Std:  {error_percentage.std():.2f}%")
        print(f" Max:  {error_percentage.max():.2f}%")
        print(f" Min:  {error_percentage.min():.2f}%")

    def plot_sMAPE_summary(self):
        """
        plot 實際值與預測值的sMAPE折線圖(1)和直方圖(2)
        """
        self.model.eval()
        all_targets = []
        all_predictions = []
        
        # 優先使用測試集，如果沒有測試集則使用驗證集
        data_loader = self.test_loader if self.test_loader is not None else self.val_loader
        dataset_name = "Test Set" if self.test_loader is not None else "Validation Set"
        filename_prefix = "test" if self.test_loader is not None else "val"
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.float().to(self.device)
                targets = targets.float().to(self.device)
                predictions = self.model(inputs)
                all_targets.append(targets.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())

        all_targets = np.concatenate(all_targets, axis=0).flatten()
        all_predictions = np.concatenate(all_predictions, axis=0).flatten()

        # calculate sMAPE
        sMAPE = []
        for i in range(len(all_targets)):
            sMAPE.append(self.sMAPE(all_targets[i], all_predictions[i]))
        sMAPE = np.array(sMAPE)

        # 1.折線圖
        plt.figure(figsize=(12, 6))
        plt.plot(sMAPE, color='purple', alpha=0.6, label='sMAPE')
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.title(f'sMAPE on {dataset_name}')
        plt.xlabel('Sample Index')
        plt.ylabel('sMAPE (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{filename_prefix}_sMAPE_line.png'))
        if self.show_plot:
            plt.show()
        plt.close()

        # 2.直方圖
        plt.figure(figsize=(8, 6))
        plt.hist(sMAPE, bins=100, color='orange', alpha=0.7, edgecolor='black')
        plt.title(f'sMAPE Distribution ({dataset_name})')
        plt.xlabel('sMAPE (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{filename_prefix}_sMAPE_histogram.png'))
        if self.show_plot:
            plt.show()
        plt.close()

        # print sMAPE statistics
        print(f"sMAPE Statistics:")
        print(f" Mean: {sMAPE.mean():.2f}%")
        print(f" Std:  {sMAPE.std():.2f}%")
        print(f" Max:  {sMAPE.max():.2f}%")
        print(f" Min:  {sMAPE.min():.2f}%")

    def load_model(self, path):
        """加載模型"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.eval()  # 確保模型處於評估模式
            print(f"模型從 {path} 加載成功")
            return checkpoint
        else:
            print(f"模型文件 {path} 不存在")
            return None
