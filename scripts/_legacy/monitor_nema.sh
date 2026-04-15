#!/bin/bash
# 实时监控 NEMA 重建进度

LOG_FILE="results/NEMA/nema_run.log"
RECON_DIR="results/NEMA/reconstructions"

echo "🔍 监控 NEMA 重建进度..."
echo ""

while true; do
    clear
    echo "======================================"
    echo "📊 NEMA 重建进度监控"
    echo "======================================"
    echo ""
    
    # 检查进程
    PROC=$(ps aux | grep -E "python.*nema_reconstruct" | grep -v grep | head -1)
    if [ -z "$PROC" ]; then
        echo "❌ 进程未运行"
        echo ""
        echo "最终日志："
        tail -50 "$LOG_FILE" 2>/dev/null || echo "日志文件不存在"
        break
    else
        PID=$(echo "$PROC" | awk '{print $2}')
        CPU=$(echo "$PROC" | awk '{print $3}')
        MEM=$(echo "$PROC" | awk '{print $4}')
        echo "✅ 进程运行中"
        echo "   PID: $PID | CPU: ${CPU}% | MEM: ${MEM}%"
        echo ""
    fi
    
    # 检查输出文件
    echo "📁 输出文件状态："
    if [ -d "$RECON_DIR" ]; then
        ls -lh "$RECON_DIR"/*.dat 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
        [ $? -ne 0 ] && echo "   (暂无 .dat 文件)"
    else
        echo "   重建目录尚未创建"
    fi
    echo ""
    
    # 检查 OSEM 日志
    echo "📋 OSEM 重建日志："
    for log in "$RECON_DIR"/NEMA_*_osem_log.txt; do
        if [ -f "$log" ]; then
            echo "   $(basename $log):"
            tail -3 "$log" 2>/dev/null | sed 's/^/      /'
        fi
    done
    echo ""
    
    # 显示最新日志
    echo "📝 最新日志（末尾 15 行）："
    echo "--------------------------------------"
    tail -15 "$LOG_FILE" 2>/dev/null | sed 's/^/  /' || echo "  (日志文件暂无内容)"
    echo "--------------------------------------"
    echo ""
    echo "按 Ctrl+C 停止监控"
    
    sleep 3
done


