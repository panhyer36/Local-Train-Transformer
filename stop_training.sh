#!/bin/bash

# 檢查是否存在PID檔案
if [ ! -f "training.pid" ]; then
    echo "未找到 training.pid 檔案"
    echo "訓練可能已經停止或未使用 run.sh 啟動"
    exit 1
fi

# 讀取PID
PID=$(cat training.pid)

# 檢查進程是否還在運行
if ps -p $PID > /dev/null 2>&1; then
    echo "正在停止訓練進程 (PID: $PID)..."
    kill $PID
    
    # 等待進程結束
    sleep 2
    
    # 再次檢查是否已停止
    if ps -p $PID > /dev/null 2>&1; then
        echo "進程未正常停止，強制終止..."
        kill -9 $PID
    fi
    
    echo "訓練已停止"
    rm training.pid
else
    echo "進程 $PID 不存在，可能已經停止"
    rm training.pid
fi 