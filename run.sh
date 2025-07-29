#!/bin/bash

# 設置腳本在遇到錯誤時停止執行
set -e

# 獲取當前時間，用於日誌檔名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="training_${TIMESTAMP}.log"

echo "開始訓練 - $(date)"
echo "日誌將保存到: $LOG_FILE"

# 使用nohup運行Python腳本，並將輸出重定向到日誌檔
nohup python3 -u train.py > "$LOG_FILE" 2>&1 &

# 保存進程ID
PID=$!
echo "訓練進程已啟動，PID: $PID"
echo "使用以下指令查看即時日誌："
echo "tail -f $LOG_FILE"
echo ""
echo "使用以下指令停止訓練："
echo "kill $PID"
echo ""
echo "查看進程狀態："
echo "ps aux | grep $PID"

# 將PID寫入檔案，方便後續停止
echo $PID > training.pid
echo "PID已保存到 training.pid" 
