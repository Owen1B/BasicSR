#!/usr/bin/env python3
"""
测试单个病人的 OSEM 重建流程
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from batch_osem_recon import *

if __name__ == "__main__":
    patient = "BianChengMing"
    
    print(f"\n{'='*70}")
    print(f"测试单个病人 OSEM 重建流程: {patient}")
    print(f"{'='*70}\n")
    
    # 测试 Original (Iter 10)
    print("\n[1/3] Original projection (Iter 10)")
    print("-" * 70)
    success, duration, message, output_file = run_osem_recon(
        patient, 'original', iterations=10, skip_if_exists=False
    )
    print(f"结果: {'✅ 成功' if success else '❌ 失败'}")
    if output_file:
        print(f"输出: {output_file}")
    print(f"耗时: {duration:.1f}s")
    
    # 测试 Denoised (Iter 10)
    print("\n[2/3] Denoised projection (Iter 10)")
    print("-" * 70)
    success, duration, message, output_file = run_osem_recon(
        patient, 'denoised', iterations=10, skip_if_exists=False
    )
    print(f"结果: {'✅ 成功' if success else '❌ 失败'}")
    if output_file:
        print(f"输出: {output_file}")
    print(f"耗时: {duration:.1f}s")
    
    # 测试 Denoised (Iter 30)
    print("\n[3/3] Denoised projection (Iter 30)")
    print("-" * 70)
    success, duration, message, output_file = run_osem_recon(
        patient, 'denoised', iterations=30, skip_if_exists=False
    )
    print(f"结果: {'✅ 成功' if success else '❌ 失败'}")
    if output_file:
        print(f"输出: {output_file}")
    print(f"耗时: {duration:.1f}s")
    
    print(f"\n{'='*70}")
    print("测试完成")
    print(f"{'='*70}\n")


