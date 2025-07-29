# 基于Transformer的时间序列预测模型

**注意: 这个项目主要用于测试和验证在每個客戶端本地上执行深度学习模型训练流程的效果与可行性。並且使用所有電器Feature**

本项目旨在为多个独立的客户（Consumer 1-50）和一栋公共建筑（Public Building）训练专门的时间序列预测模型。每个模型都基于Transformer架构，能够从历史数据中学习模式并预测未来的数值。项目采用配置化架构，支持灵活的特征选择和数据标准化。

## ✨ 项目特点

- **配置化架构**: 通过 `config.yaml` 文件统一管理所有模型、数据和训练参数
- **批量训练**: `train.py` 脚本可自动为所有数据集逐一训练模型
- **后台执行**: `run.sh` 脚本支持在后台启动训练，并将日志保存到文件，不影响终端的正常使用
- **高级模型架构**: 采用改进的Transformer Encoder架构，使用GELU激活函数、LayerNorm和Xavier初始化
- **智能数据预处理**: 支持MinMaxScaler和StandardScaler数据标准化，自动保存和加载标准化器
- **可视化输出**: 训练完成后，会自动生成训练历史、预测结果对比、注意力权重分布、误差分析等多种图表
- **模块化设计**: 分为数据加载 (`DataLoader.py`)、模型定义 (`Model.py`) 和训练逻辑 (`Trainer.py`)
- **自动设备选择**: 支持CUDA、MPS和CPU自动选择
- **错误处理**: 完善的错误处理和进度显示机制

## 📁 项目结构

```
.
├── data/                  # 数据目录
│   └── processed/         # 存放预处理后的CSV文件
├── runs/                  # 所有训练产出的根目录
│   ├── Consumer_01/       # 单个模型的训练结果
│   │   ├── transformer_model.pth     # 保存的训练好的模型
│   │   ├── training_history.png      # 训练/验证损失曲线
│   │   ├── val_predictions.png       # 验证集预测结果图
│   │   ├── error_analysis.png        # 误差百分比分析图
│   │   └── ...            # 其他可视化图表
│   └── ...
├── scalers/               # 数据标准化器保存目录
│   ├── Consumer_01_feature_scaler.pkl
│   ├── Consumer_01_target_scaler.pkl
│   └── ...
├── src/                   # 源代码目录
│   ├── DataLoader.py      # 数据加载和预处理
│   ├── Model.py           # Transformer模型定义
│   └── Trainer.py         # 训练、验证和评估逻辑
├── config.yaml            # 主配置文件
├── config.py              # 配置管理模块
├── requirements.txt       # Python依赖包
├── train.py               # 主训练脚本
├── run.sh                 # 启动后台训练的脚本
└── stop_training.sh       # 停止后台训练的脚本
```

## ⚙️ 环境搭建

1.  **下載此項目**

2.  **创建虚拟环境 (推荐)**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

## 📊 数据准备

将所有预处理好的 `.csv` 数据文件放入 `data/processed/` 目录下。脚本将按以下命名约定查找文件：
- `Consumer_01.csv`, `Consumer_02.csv`, ..., `Consumer_50.csv`
- `Public_Building.csv`

### 数据特征
- 支持17个特征维度的CSV文件
- 自动移除 'Unnamed: 0' 字段
- 支持特征选择和数据标准化
- 分别对特征和目标进行标准化处理

## 🔧 配置系统

项目采用配置化架构，所有参数都在 `config.yaml` 中定义：

### 模型参数
- **feature_dim**: 17 (数据特征维度)
- **d_model**: 256 (Transformer隐藏层维度)
- **nhead**: 8 (多头注意力头数)
- **num_layers**: 4 (Transformer层数)
- **dropout**: 0.1 (Dropout比率)

### 数据配置
- **seq_len**: 96 (输入序列长度)
- **normalize**: true (是否标准化)
- **scaler_type**: "minmax" (标准化类型)
- **features**: 可选的特征列表
- **target_feature**: "Power_Demand" (目标特征)

### 训练参数
- **batch_size**: 32
- **learning_rate**: 1e-4
- **train_ratio**: 0.8
- **early_stopping_patience**: 10
- **num_epochs**: 100
- **weight_decay**: 1e-5

## 🚀 如何使用

### 训练模型

1.  **前台训练**
    ```bash
    python3 train.py
    ```

2.  **后台训练**
    执行 `run.sh` 脚本来启动所有模型的批量训练。该脚本会在后台运行 `train.py`。
    ```bash
    bash run.sh
    ```
    启动后，终端会显示训练进程的PID，并提示日志文件的路径。

3.  **停止训练**
    执行 `stop_training.sh` 脚本可以安全地终止正在后台运行的训练进程。
    ```bash
    bash stop_training.sh
    ```
    或者，你也可以使用 `kill` 命令手动停止：
    ```bash
    kill $(cat training.pid)
    ```

### 监控训练进度

1.  **查看训练日志**
    ```bash
    tail -f training_*.log
    ```

2.  **查看训练进程**
    ```bash
    ps aux | grep $(cat training.pid)
    ```

### 配置管理

1.  **查看配置**
    ```bash
    python3 config.py
    ```

2.  **修改配置**
    编辑 `config.yaml` 文件来调整模型、数据和训练参数。

## 📈 输出说明

每个模型的训练结果都保存在 `runs/` 下的对应文件夹中（例如 `runs/Consumer_01/`）。主要输出文件包括：

- `transformer_model.pth`: 训练好的PyTorch模型文件
- `training_history.png`: 展示训练过程中的训练损失和验证损失的变化曲线
- `val_predictions.png`: 在验证集上，模型预测值与真实值的对比图
- `perfect_prediction.png`: 理想情况下的完美预测图，用于对比
- `attention_weights.png`: 抽样展示模型在进行预测时，对输入序列中不同时间步的注意力权重分布
- `average_attention_weights.png`: 模型在整个验证集上的平均注意力权重分布，反映了模型普遍关注的时间模式
- `error_analysis.png`: 误差百分比分析图表，展示预测误差的分布情况

### 标准化器文件
标准化器会自动保存在 `scalers/` 目录中：
- `{客户端名称}_feature_scaler.pkl`: 特征标准化器
- `{客户端名称}_target_scaler.pkl`: 目标标准化器

## 🏗️ 核心架构

### 模型架构
- **Transformer模型**: 改进的Encoder-only架构，使用GELU激活函数
- **时间序列预测**: 使用前96个时间步预测下一个时间步的Power_Demand值
- **注意力机制**: 改进的注意力聚合机制，使用多层投影网络
- **位置编码**: 更好的位置编码实现

### 主要组件

1. **DataLoader** (`src/DataLoader.py`): 
   - 支持MinMaxScaler和StandardScaler数据标准化
   - 自动保存和加载标准化器到scalers目录
   - 返回(input_sequence, target_value)格式的数据
   - 支持配置化的特征选择

2. **Model** (`src/Model.py`): 
   - 改进的TransformerModel类，使用GELU激活函数
   - 更好的位置编码实现
   - 多层注意力聚合机制

3. **Trainer** (`src/Trainer.py`): 
   - 自动设备选择(CUDA/MPS/CPU)
   - 支持tqdm进度条
   - 可选的TensorBoard日志记录
   - 新增误差百分比分析图表

4. **train.py**: 配置化的主训练脚本，支持错误处理

## 📦 依赖环境

核心依赖在 `requirements.txt` 中定义：
- **torch>=2.0.0**: 深度学习框架
- **pandas>=1.5.0**: 数据处理
- **scikit-learn>=1.1.0**: 数据标准化
- **matplotlib>=3.5.0**: 可视化
- **pyyaml>=6.0**: 配置文件
- **tqdm>=4.64.0**: 进度条
- **psutil>=5.9.0**: 系统监控

## 🔍 特殊说明

### 数据预处理
- 自动移除 'Unnamed: 0' 字段
- 支持特征选择和数据标准化
- 分别对特征和目标进行标准化
- 标准化器自动保存为 `{客户端名称}_feature_scaler.pkl` 和 `{客户端名称}_target_scaler.pkl`

### 模型特点
- 改进的注意力聚合机制
- 支持多种激活函数(GELU)
- 包含LayerNorm和Xavier初始化
- 自动设备选择和梯度裁剪

### 训练流程
1. 从config.yaml加载配置
2. 为每个数据文件创建独立的模型和标准化器
3. 支持错误处理和进度显示
4. 生成多种可视化图表
5. 支持TensorBoard日志记录

### 配置化特性
- 所有参数都可通过config.yaml配置
- 支持特征选择和数据标准化选项
- 灵活的输出目录和日志配置
- 实验管理和版本控制 