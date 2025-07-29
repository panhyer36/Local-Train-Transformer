import os
import glob
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import psutil


class SequenceCSVDataset(Dataset):
    def __init__(self, file_name, config=None, seq_len=96, target_len=1, 
                 normalize=True, scaler_type="minmax", features=None, 
                 target_feature="Power_Demand", train_indices=None):
        """
        初始化序列數據集
        
        Args:
            file_name: CSV文件路径
            config: 配置对象（可选）
            seq_len: 輸入序列長度
            target_len: 目標序列長度
            normalize: 是否標準化數據
            scaler_type: 標準化類型 ('minmax' 或 'standard')
            features: 要使用的特徵列表（可選）
            target_feature: 目標特徵名稱
            train_indices: 訓練集的序列索引（用於確保標準化器僅在訓練集上擬合）
        """
        # 如果提供了config，使用config中的參數
        if config is not None:
            seq_len = config.data.seq_len
            target_len = config.data.target_len
            normalize = config.data.normalize
            scaler_type = config.data.scaler_type
            features = config.data.features
            target_feature = config.data.target_feature
        
        self.seq_len = seq_len
        self.target_len = target_len
        self.normalize = normalize
        self.scaler_type = scaler_type
        self.target_feature = target_feature
        self.file_name = file_name
        
        # 轉換為絕對路徑，確保路徑正確
        abs_file_path = os.path.abspath(file_name)
        self.files = abs_file_path
        
        # 讀取CSV文件
        df = pd.read_csv(self.files)
        
        # 移除未命名列
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        # 選擇特徵
        if features is not None:
            # 確保目標特徵在特徵列表中
            if target_feature not in features:
                features.append(target_feature)
            # 只選擇指定的特徵
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                print(f"警告: 以下特徵在數據中不存在: {missing_features}")
                features = [f for f in features if f in df.columns]
            df = df[features]
        
        # 創建序列（先創建序列再進行標準化）
        self.sequences = []
        self.targets = []
        
        # 創建輸入序列和目標值
        total_len = seq_len + target_len
        for i in range(len(df) - total_len + 1):
            # 輸入序列（前seq_len個時間步）
            input_seq = df.iloc[i:i+seq_len].values
            # 目標值（後target_len個時間步的目標特徵）
            target_seq = df.iloc[i+seq_len:i+seq_len+target_len][target_feature].values
            
            self.sequences.append(input_seq)
            self.targets.append(target_seq)
        
        # 數據標準化（僅在訓練集序列上擬合）
        if normalize:
            self.feature_scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
            self.target_scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
            
            # 獲取特徵列和目標列索引
            feature_cols = [i for i, col in enumerate(df.columns) if col != target_feature]
            
            if train_indices is not None:
                # 僅使用訓練集序列來擬合標準化器
                train_sequences = [self.sequences[i] for i in train_indices]
                train_targets = [self.targets[i] for i in train_indices]
                
                # 合併所有訓練序列的特徵數據來擬合特徵標準化器
                if len(feature_cols) > 0:
                    train_features = np.vstack([seq[:, feature_cols] for seq in train_sequences])
                    self.feature_scaler.fit(train_features)
                
                # 合併所有訓練目標來擬合目標標準化器
                train_targets_flat = np.concatenate(train_targets).reshape(-1, 1)
                self.target_scaler.fit(train_targets_flat)
                
                print(f"標準化器已在{len(train_indices)}個訓練序列上擬合")
            else:
                # 如果沒有提供train_indices，在整個數據集上擬合（向後兼容）
                print("警告: 沒有提供train_indices，將在整個數據集上擬合標準化器，可能導致數據洩漏")
                if len(feature_cols) > 0:
                    all_features = np.vstack([seq[:, feature_cols] for seq in self.sequences])
                    self.feature_scaler.fit(all_features)
                
                all_targets = np.concatenate(self.targets).reshape(-1, 1)
                self.target_scaler.fit(all_targets)
            
            # 標準化所有序列
            for i in range(len(self.sequences)):
                if len(feature_cols) > 0:
                    self.sequences[i][:, feature_cols] = self.feature_scaler.transform(self.sequences[i][:, feature_cols])
                self.targets[i] = self.target_scaler.transform(self.targets[i].reshape(-1, 1)).flatten()
            
            # 保存scaler
            client_name = os.path.basename(file_name).replace('.csv', '')
            self.save_scaler(client_name)
        else:
            self.feature_scaler = None
            self.target_scaler = None
        
        # 獲取目標特徵的列索引
        self.target_idx = df.columns.get_loc(target_feature)
        
        # 轉換為numpy數組
        self.sequences = [np.array(seq, dtype=np.float32) for seq in self.sequences]
        self.targets = [np.array(target, dtype=np.float32) for target in self.targets]
        
        print(f"{file_name}: 成功載入{len(self.sequences)}個序列")
        print(f"  輸入序列形狀: {np.array(self.sequences).shape}")
        print(f"  目標序列形狀: {np.array(self.targets).shape}")
        print(f"  特徵維度: {df.shape[1]}")
        print(f"  是否標準化: {normalize}")
        
        process = psutil.Process(os.getpid())
        print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

    def save_scaler(self, client_name):
        """保存標準化器"""
        if self.normalize:
            scaler_dir = "scalers"
            os.makedirs(scaler_dir, exist_ok=True)
            
            # 保存特徵scaler
            if self.feature_scaler is not None:
                feature_scaler_path = os.path.join(scaler_dir, f"{client_name}_feature_scaler.pkl")
                with open(feature_scaler_path, 'wb') as f:
                    pickle.dump(self.feature_scaler, f)
            
            # 保存目標scaler
            if self.target_scaler is not None:
                target_scaler_path = os.path.join(scaler_dir, f"{client_name}_target_scaler.pkl")
                with open(target_scaler_path, 'wb') as f:
                    pickle.dump(self.target_scaler, f)
                    
            print(f"標準化器已保存到 {scaler_dir}/{client_name}_*_scaler.pkl")
    
    def load_scaler(self, client_name):
        """加載標準化器"""
        scaler_dir = "scalers"
        
        feature_scaler_path = os.path.join(scaler_dir, f"{client_name}_feature_scaler.pkl")
        target_scaler_path = os.path.join(scaler_dir, f"{client_name}_target_scaler.pkl")
        
        if os.path.exists(feature_scaler_path):
            with open(feature_scaler_path, 'rb') as f:
                self.feature_scaler = pickle.load(f)
        
        if os.path.exists(target_scaler_path):
            with open(target_scaler_path, 'rb') as f:
                self.target_scaler = pickle.load(f)
                
        print(f"標準化器已從 {scaler_dir}/{client_name}_*_scaler.pkl 加載")
    
    def inverse_transform_target(self, target):
        """反標準化目標值"""
        if self.target_scaler is not None:
            # 確保shape正確
            if target.ndim == 1:
                target = target.reshape(-1, 1)
            return self.target_scaler.inverse_transform(target)
        return target
    
    def inverse_transform_features(self, features):
        """反標準化特徵"""
        if self.feature_scaler is not None:
            return self.feature_scaler.inverse_transform(features)
        return features

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        返回輸入序列和目標值
        
        Returns:
            tuple: (input_sequence, target_value)
                - input_sequence: shape (seq_len, feature_dim)
                - target_value: shape (target_len,) 或 scalar
        """
        input_seq = self.sequences[idx]  # shape: (seq_len, feature_dim)
        target = self.targets[idx]  # shape: (target_len,)
        
        # 如果target_len為1，返回scalar
        if self.target_len == 1:
            target = target[0]
        
        return input_seq, target


def create_sequential_datasets(dataset_or_file, train_ratio=0.8, seq_len=96, target_len=1):
    """
    將數據集按順序拆分為訓練集和驗證集（適用於時間序列）
    
    Args:
        dataset_or_file: 數據集對象或CSV文件路徑
        train_ratio: 訓練集比例
        seq_len: 序列長度（當傳入文件路徑時需要）
        target_len: 目標長度（當傳入文件路徑時需要）
    
    Returns:
        tuple: (train_indices, val_indices)
    """
    # 如果傳入的是字符串（文件路徑），直接從文件計算序列數量
    if isinstance(dataset_or_file, str):
        df = pd.read_csv(dataset_or_file)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        # 計算可以創建的序列數量
        total_len = seq_len + target_len
        total_size = len(df) - total_len + 1
        
        if total_size <= 0:
            raise ValueError(f"數據長度 {len(df)} 不足以創建長度為 {seq_len + target_len} 的序列")
    else:
        # 如果傳入的是數據集對象，使用其長度
        total_size = len(dataset_or_file)
    
    train_size = int(train_ratio * total_size)
    
    # 順序拆分：前train_size個作為訓練集，其餘作為驗證集
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))
    
    print(f"順序拆分: 訓練集 {len(train_indices)} 個序列, 驗證集 {len(val_indices)} 個序列")
    
    return train_indices, val_indices

