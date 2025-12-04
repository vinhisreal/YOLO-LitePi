#!/bin/bash

echo "Optimizing Raspberry Pi 5 system for maximum performance..."

# 1) Clear Linux caches

echo "Clearing Linux caches..."
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

# 2) Set CPU governor to performance
echo "Setting CPU governor to performance..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
do
    echo performance | sudo tee $cpu > /dev/null
done

# 3) Reduce swappiness
echo "Set swappiness = 10"
echo 10 | sudo tee /proc/sys/vm/swappiness > /dev/null

# 4) Clear Python cache
echo "Clearing Python __pycache__..."
find . -type d -name "__pycache__" -exec rm -rf {} +

# 5) Export optimized environment variables

echo "Exporting environment variables for max performance..."

# Disable OpenCV multithread (để không bị conflict với PyTorch)
export OPENCV_FOR_THREADS_NUM=4

# PyTorch optimizations
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# NCNN optim (để Pi 5 chạy ổn định hơn)
export NCNN_THREADS=4

# Python optim
export PYTHONHASHSEED=0

# 6) Optional: bind CPU core affinity

echo "🚀 Starting End-to-End Pipeline..."
python3 e2e_optimize.py

echo "🏁 Done."
