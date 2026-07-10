#!/bin/bash
# 自动清理缓存并编译 GroundingDINO CUDA 算子的脚本
echo "Cleaning old build files..."
rm -rf build/
rm -rf *.egg-info/

echo "Building CUDA extensions..."
python setup.py build_ext --inplace

echo "Build complete! Check if groundingdino/_C.*.so exists."
