#!/usr/bin/env python
"""
IndexTTS2 模型下载脚本
自动下载所需的模型文件
"""
import os
import sys

# 确保在项目目录
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

from indextts.utils.model_download import ensure_models_available


def main():
    """下载所有必需的模型"""
    model_dir = "checkpoints"

    print("=" * 60)
    print("IndexTTS2 模型下载")
    print("=" * 60)
    print(f"模型目录: {os.path.abspath(model_dir)}")
    print()

    # 自动下载缺失的模型
    ensure_models_available(model_dir)

    print()
    print("=" * 60)
    print("模型下载完成!")
    print("=" * 60)

    # 列出已下载的文件
    print("\n已下载的模型文件:")
    for f in os.listdir(model_dir):
        path = os.path.join(model_dir, f)
        if os.path.isfile(path):
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"  {f}: {size:.1f} MB")


if __name__ == "__main__":
    main()